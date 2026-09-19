#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""自己控制量化范围：只量化指定算子（白名单）+ 真实数据校准。

动机：全局 int8 会把循环状态路径（attention 的 k/v 投影）一起压掉，
每层状态误差 2~10%，32 层递归后模型失忆。这里改成只压 FFN 两个大矩阵
（占权重 2/3），把 attention/wkv 状态路径留给 fp16。

用法:
  python quantize_layer_sel.py <输入.onnx> <输出.onnx> \
      --calib calib/layer07.npz --include "/ffn/key/MatMul,/ffn/value/MatMul"
  --include 支持逗号分隔的子串匹配；不写则退回全图量化。
"""

import argparse
import os
import sys
import time

sys.path.insert(0, "/home/disk/cann80base/ascend-toolkit/latest/tools")
sys.path.insert(0, "/home/disk/models/rwkv7-2.9b")

import onnx                   # noqa: E402

from modelslim.onnx.post_training_quant import run_quantize, QuantConfig  # noqa: E402
from quantize_layer import fix_axes_to_attribute, toposort_nodes          # noqa: E402
from quantize_layer_calib import build_calib_data                        # noqa: E402


def expand_include(model, patterns):
    """把子串模式展开成具体的 MatMul 节点名。"""
    if not patterns:
        return None
    wanted = [p.strip() for p in patterns.split(",") if p.strip()]
    names = [n.name for n in model.graph.node
             if n.op_type == "MatMul" and any(p in n.name for p in wanted)]
    return names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument("--calib", required=True)
    parser.add_argument("--include", default="",
                        help="只量化名字含这些子串的 MatMul（逗号分隔）")
    parser.add_argument("--exclude-lora", action="store_true", default=True)
    parser.add_argument("--method", type=int, default=1)
    args = parser.parse_args()

    tmp = (args.dst[:-5] if args.dst.endswith(".onnx") else args.dst) + "_raw.onnx"
    pre = onnx.load(args.src, load_external_data=False)

    quantize_nodes = expand_include(pre, args.include)
    if quantize_nodes is not None:
        print("白名单量化节点 %d 个：%s" % (len(quantize_nodes), quantize_nodes), flush=True)

    calib_data = build_calib_data(pre, args.calib)
    print("校准样本 %d 组，输入 %d 个" % (len(calib_data), len(calib_data[0])), flush=True)

    t0 = time.time()
    run_quantize(args.src, tmp,
                 QuantConfig(quant_mode=1, is_per_channel=True, calib_data=calib_data,
                             calib_method=args.method, quantize_nodes=quantize_nodes or [],
                             is_optimize_graph=False))
    print("量化完成 %.1f s" % (time.time() - t0), flush=True)

    model = onnx.load(tmp, load_external_data=False)
    print("修复 axes 节点 %d 个" % fix_axes_to_attribute(model), flush=True)
    ordered, stuck = toposort_nodes(model)
    print("拓扑排序 %d 个（未排序 %d）" % (ordered, stuck), flush=True)
    onnx.save(model, args.dst, save_as_external_data=False)
    os.remove(tmp)

    import collections
    counter = collections.Counter(n.op_type for n in model.graph.node)
    print("最终算子（前 6）: %s" % dict(counter.most_common(6)))
    print("输出 %s（%.1f MB）" % (args.dst, os.path.getsize(args.dst) / 1e6))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
