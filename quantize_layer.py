#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对单层 ONNX 做完整量化流水线（供 ATC 编译）：

  1) modelslim 免校准量化（label-free PTQ，int8，per-channel）
  2) 修复 opset：modelslim 会把 opset 标成 11，但 Unsqueeze/Squeeze 仍是
     opset13+ 的"axes 作为输入"形式 → 改成 opset11 的属性形式

用法: python quantize_layer.py <输入.onnx> <输出.onnx>
"""

import os
import re
import sys
import time

sys.path.insert(0, "/home/disk/cann80base/ascend-toolkit/latest/tools")

import numpy as np            # noqa: E402
import onnx                   # noqa: E402
from onnx import helper, numpy_helper  # noqa: E402

from modelslim.onnx.post_training_quant import run_quantize, QuantConfig  # noqa: E402


def fix_axes_to_attribute(model):
    """把 axes 从"输入"改回"属性"（opset 11 语义）。

    注意 ReduceSum 也是同一类：opset 13 起 axes 变成输入，opset 11 是属性。
    """
    axes_input_ops = ("Unsqueeze", "Squeeze", "ReduceSum")
    inits = {i.name: i for i in model.graph.initializer}
    fixed = 0
    for node in model.graph.node:
        if node.op_type in axes_input_ops and len(node.input) == 2:
            axes_name = node.input[1]
            if axes_name not in inits:
                continue
            axes = np.asarray(numpy_helper.to_array(inits[axes_name])).reshape(-1).tolist()
            del node.input[1]
            node.attribute.extend([helper.make_attribute("axes", [int(a) for a in axes])])
            fixed += 1
    return fixed


def toposort_nodes(model):
    """modelslim 输出的节点顺序不是拓扑序，这里重排（否则 onnx.checker / ATC 会拒绝）。"""
    produced = {i.name for i in model.graph.input} | {i.name for i in model.graph.initializer}
    remaining = list(model.graph.node)
    ordered = []
    while remaining:
        rest = []
        progressed = False
        for node in remaining:
            if all((x in produced) or x == "" for x in node.input):
                ordered.append(node)
                produced.update(node.output)
                progressed = True
            else:
                rest.append(node)
        remaining = rest
        if not progressed:
            break
    del model.graph.node[:]
    model.graph.node.extend(ordered + remaining)
    return len(ordered), len(remaining)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    tmp = (dst[:-5] if dst.endswith(".onnx") else dst) + "_raw.onnx"

    # LoRA 的小矩阵量化后会让 TBE 的 FixPipe 单算子编译失败（"fix_pipe does not support
    # single op compilation"），而它们只占参数量 4%，直接排除；大矩阵(96%)照常量化。
    pre = onnx.load(src, load_external_data=False)
    exclude = [n.name for n in pre.graph.node
               if n.op_type == "MatMul" and re.match(r"^/att/MatMul(_\d+)?$", n.name)]
    print("排除量化的小矩阵节点 %d 个: %s" % (len(exclude), exclude), flush=True)

    print("量化 %s" % src, flush=True)
    t0 = time.time()
    run_quantize(src, tmp, QuantConfig(is_optimize_graph=False, exclude_nodes=exclude))
    print("量化完成 %.1f s，%.1f MB → %.1f MB"
          % (time.time() - t0, os.path.getsize(src) / 1e6, os.path.getsize(tmp) / 1e6))

    model = onnx.load(tmp, load_external_data=False)
    fixed = fix_axes_to_attribute(model)
    print("修复 Unsqueeze/Squeeze 节点数:", fixed)
    ordered, stuck = toposort_nodes(model)
    print("拓扑排序: 已排序 %d 个节点，未能排序 %d 个" % (ordered, stuck))
    # 注意：不要调用 onnx.checker —— 昇腾自定义算子(AscendQuant/AscendDequant)
    # 不在 ONNX 标准 schema 内，校验必然失败；交给 ATC 自己校验即可。
    onnx.save(model, dst, save_as_external_data=False)
    os.remove(tmp)

    import collections
    counter = collections.Counter(n.op_type for n in model.graph.node)
    print("最终算子（前 8）:", dict(counter.most_common(8)))
    print("输出 %.1f MB" % (os.path.getsize(dst) / 1e6))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
