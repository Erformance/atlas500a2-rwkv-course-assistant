#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P3-W 前置探测：量化后的 ONNX 里 int8 权重是怎么存的？

只有当量化图里能稳定取出"反量化后的权重"时，才能算协议 §5 要求的候选指标之一
"权重 MSE"。本脚本对一份已存在的量化图（layer31_cp_q.onnx）与对应的 fp16 简化图
（layer31_sim.onnx）做结构比对，报告：
  1) 两张图的算子构成；
  2) int8/uint8 初始化器（int8 权重本体）的名字、形状；
  3) QuantizeLinear / DequantizeLinear 节点的 scale / zero_point 名；
  4) 两张图同名的权重张量有多少（能否逐张量配对）。
"""

import collections
import os
import sys

import onnx
from onnx import numpy_helper

MODEL_DIR = os.environ.get("RWKV_MODEL_DIR", "/home/disk/models/rwkv7-2.9b")


def summary(path, tag):
    print("\n===== %s：%s" % (tag, os.path.basename(path)), flush=True)
    if not os.path.exists(path):
        print("  文件不存在", flush=True)
        return None
    m = onnx.load(path, load_external_data=False)
    ops = collections.Counter(n.op_type for n in m.graph.node)
    print("  节点 %d，算子：%s" % (len(m.graph.node), dict(ops.most_common(12))), flush=True)
    inits = {t.name: t for t in m.graph.initializer}
    print("  初始化器 %d 个" % len(inits), flush=True)
    dtypes = collections.Counter(t.data_type for t in inits.values())
    print("  dtype 分布：%s" % {int(k): v for k, v in dtypes.most_common()}, flush=True)
    quant = [(n, t) for n, t in inits.items()
             if t.data_type in (onnx.TensorProto.INT8, onnx.TensorProto.UINT8)]
    for name, t in quant[:6]:
        arr = numpy_helper.to_array(t)
        print("    int8 张量 %-40s shape=%s dtype=%s"
              % (name, arr.shape, arr.dtype), flush=True)
    if len(quant) > 6:
        print("    …共 %d 个整型张量" % len(quant), flush=True)
    for node in m.graph.node:
        if node.op_type in ("QuantizeLinear", "DequantizeLinear"):
            print("    %s -> %s 输入 %s"
                  % (node.op_type, list(node.output), list(node.input)), flush=True)
            if len([n for n in m.graph.node
                    if n.op_type in ("QuantizeLinear", "DequantizeLinear")]) > 8:
                print("    （该图 Q/DQ 节点较多，上面只示意前若干条）", flush=True)
                break
    return inits


def main():
    q = summary(os.path.join(MODEL_DIR, "layer31_cp_q.onnx"), "量化图")
    f = summary(os.path.join(MODEL_DIR, "layer31_sim.onnx"), "fp16 简化图")
    if q and f:
        shared = [n for n in q if n in f]
        same_shape = sum(1 for n in shared
                         if list(q[n].dims) == list(f[n].dims))
        print("\n同名张量 %d 个（形状相同 %d 个）" % (len(shared), same_shape), flush=True)
        print("样例：%s" % shared[:8], flush=True)
    print("\n=== 候选：DequantizeLinear 的 scale 是否在初始化器里 ===", flush=True)
    if q:
        m = onnx.load(os.path.join(MODEL_DIR, "layer31_cp_q.onnx"), load_external_data=False)
        names = []
        for node in m.graph.node:
            if node.op_type == "DequantizeLinear" and len(node.input) >= 2:
                names.append(node.input[1])
        print("scale 名 %d 个，前 6：%s" % (len(names), names[:6]), flush=True)
        hit = [n for n in names if n in q]
        print("在初始化器里能找到的 scale：%d/%d" % (len(hit), len(names)), flush=True)

    print("\n=== AscendQuant / AscendDequant 节点的完整输入 ===", flush=True)
    if q:
        m = onnx.load(os.path.join(MODEL_DIR, "layer31_cp_q.onnx"), load_external_data=False)
        n_show = 0
        for node in m.graph.node:
            if node.op_type in ("AscendQuant", "AscendDequant"):
                n_show += 1
                if n_show <= 14:
                    print("  %-15s in=%s out=%s" % (node.op_type, list(node.input),
                                                     list(node.output)), flush=True)
        print("  共 %d 个 Ascend 量化节点" % n_show, flush=True)
        print("  int8 初始化器名：%s" % [n for n, t in q.items()
                                         if t.data_type == onnx.TensorProto.INT8], flush=True)
        print("  FLOAT 初始化器名（前 12）：%s"
              % [n for n, t in q.items() if t.data_type == onnx.TensorProto.FLOAT][:12],
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
