#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""消除 RWKV ONNX 里的 DequantizeLinear 节点（CANN 8.0.RC1 的 ONNX 解析器不支持）。

做法：把被量化的嵌入表离线反量化成 fp32（同一初始器名字），把 Gather 的输出
改名成原 DequantizeLinear 的输出名，然后删掉 DequantizeLinear 节点。

用法：python rwkv_onnx_fix_dequant.py model_clean.onnx model_fixed.onnx
"""

import argparse

import numpy as np
import onnx
from onnx import numpy_helper


def get_initializer_map(graph):
    return {init.name: init for init in graph.initializer}


def read_tensor(graph, inits, name):
    if name in inits:
        return numpy_helper.to_array(inits[name]), "initializer", inits[name]
    for node in graph.node:
        if node.op_type == "Constant" and name in list(node.output):
            for attr in node.attribute:
                if attr.name == "value":
                    return numpy_helper.to_array(attr.t), "constant", node
    raise KeyError("找不到张量 %s" % name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src")
    parser.add_argument("dst")
    args = parser.parse_args()

    model = onnx.load(args.src)
    graph = model.graph
    inits = get_initializer_map(graph)

    dq_nodes = [n for n in graph.node if n.op_type == "DequantizeLinear"]
    if not dq_nodes:
        print("没有 DequantizeLinear 节点，直接另存")
        onnx.save(model, args.dst)
        return 0
    if len(dq_nodes) != 1:
        raise SystemExit("预期 1 个 DequantizeLinear，实际 %d 个" % len(dq_nodes))

    node = dq_nodes[0]
    quant_name, scale_name, zp_name = list(node.input)[:3]
    out_name = list(node.output)[0]
    print("DequantizeLinear: %s -> %s" % (quant_name, out_name))

    scale_arr, _, _ = read_tensor(graph, inits, scale_name)
    zp_arr, _, _ = read_tensor(graph, inits, zp_name)
    scale = float(np.asarray(scale_arr).reshape(-1)[0])
    zero_point = float(np.asarray(zp_arr).reshape(-1)[0])
    print("scale=%r zero_point=%r" % (scale, zero_point))

    producers = [n for n in graph.node if quant_name in list(n.output)]
    if len(producers) != 1:
        raise SystemExit("量化张量有 %d 个生产者" % len(producers))
    producer = producers[0]
    print("生产者: %s (%s) 输入=%s" % (producer.name, producer.op_type, list(producer.input)))
    if producer.op_type != "Gather":
        raise SystemExit("预期生产者是 Gather，实际是 %s" % producer.op_type)

    table_name = list(producer.input)[0]
    table, kind, holder = read_tensor(graph, inits, table_name)
    print("量化表: %s (%s) dtype=%s shape=%s" % (table_name, kind, table.dtype, table.shape))

    table_f32 = (table.astype(np.float32) - zero_point) * scale
    print("反量化后: dtype=%s shape=%s 范围=[%.4f, %.4f]"
          % (table_f32.dtype, table_f32.shape, float(table_f32.min()), float(table_f32.max())))

    if kind == "initializer":
        new_init = numpy_helper.from_array(table_f32, name=table_name)
        idx = list(graph.initializer).index(holder)
        del graph.initializer[idx]
        graph.initializer.insert(idx, new_init)
    else:
        for attr in holder.attribute:
            if attr.name == "value":
                attr.t.CopyFrom(numpy_helper.from_array(table_f32, name=table_name))

    assert list(producer.output)[0] == quant_name
    producer.output[0] = out_name
    graph.node.remove(node)

    del graph.value_info[:]
    onnx.checker.check_model(model)
    onnx.save(model, args.dst)
    print("已保存 %s" % args.dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
