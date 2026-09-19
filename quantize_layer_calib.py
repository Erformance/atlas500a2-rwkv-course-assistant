#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""带真实校准数据的 int8 量化（quantize_layer.py 的加强版）。

和 label-free 版的区别只有一处：把 capture_calib.py 抓到的真实输入喂给
modelslim 做激活值范围统计，避免深层激活被默认刻度截断。

用法:
  python quantize_layer_calib.py <输入.onnx> <输出.onnx> --calib calib/layer07.npz [--method 0]
  --method: 0 min-max（默认）, 1 percentile, 2 entropy
"""

import argparse
import os
import re
import sys
import time

sys.path.insert(0, "/home/disk/cann80base/ascend-toolkit/latest/tools")
sys.path.insert(0, "/home/disk/models/rwkv7-2.9b")

import numpy as np            # noqa: E402
import onnx                   # noqa: E402

from modelslim.onnx.post_training_quant import run_quantize, QuantConfig  # noqa: E402
from quantize_layer import fix_axes_to_attribute, toposort_nodes  # noqa: E402


def input_specs(model):
    specs = []
    for vi in model.graph.input:
        dims = []
        for d in vi.type.tensor_type.shape.dim:
            dims.append(d.dim_value if d.dim_value > 0 else 1)
        specs.append((vi.name, dims))
    return specs


def build_calib_data(model, npz_path):
    data = np.load(npz_path)
    keys = set(data.files)
    series = []
    for name, dims in input_specs(model):
        key = re.sub(r"\.\d+$", "", name.split(":")[-1])
        if key not in keys:
            raise KeyError("校准数据里缺少输入 %s（现有 %s）" % (key, sorted(keys)))
        arr = data[key]
        series.append((key, arr.reshape(arr.shape[0], -1)))
    steps = series[0][1].shape[0]
    for key, arr in series:
        if arr.shape[0] != steps:
            raise ValueError("校准样本数不一致：%s" % key)
    dims_list = [d for _, d in input_specs(model)]
    samples = []
    for t in range(steps):
        sample = []
        for (key, flat), dims in zip(series, dims_list):
            sample.append(flat[t].reshape(dims).astype(np.float32))
        samples.append(sample)
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument("--calib", required=True, help="capture_calib.py 产出的 npz")
    parser.add_argument("--method", type=int, default=0, help="0 min-max, 1 percentile, 2 entropy")
    args = parser.parse_args()

    tmp = (args.dst[:-5] if args.dst.endswith(".onnx") else args.dst) + "_raw.onnx"
    pre = onnx.load(args.src, load_external_data=False)
    exclude = [n.name for n in pre.graph.node
               if n.op_type == "MatMul" and re.match(r"^/att/MatMul(_\d+)?$", n.name)]
    calib_data = build_calib_data(pre, args.calib)
    print("校准样本 %d 组，输入 %d 个；排除 LoRA 小矩阵 %d 个"
          % (len(calib_data), len(calib_data[0]), len(exclude)), flush=True)

    t0 = time.time()
    run_quantize(args.src, tmp,
                 QuantConfig(quant_mode=1, is_per_channel=True, calib_data=calib_data,
                             calib_method=args.method, exclude_nodes=exclude,
                             is_optimize_graph=False))
    print("量化完成 %.1f s" % (time.time() - t0), flush=True)

    model = onnx.load(tmp, load_external_data=False)
    print("修复 axes 节点 %d 个" % fix_axes_to_attribute(model), flush=True)
    ordered, stuck = toposort_nodes(model)
    print("拓扑排序 %d 个（未排序 %d）" % (ordered, stuck), flush=True)
    onnx.save(model, args.dst, save_as_external_data=False)
    os.remove(tmp)
    print("输出 %s（%.1f MB）" % (args.dst, os.path.getsize(args.dst) / 1e6))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
