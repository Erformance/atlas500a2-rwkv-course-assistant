#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P3-W：协议 §5 候选指标里缺的那一个——"权重 MSE"。

背景：协议 §5 要求比较六类敏感度，其中"权重 MSE"此前未做，原因是 int8 ONNX 在 ATC 编译后
被删除（见 P3 报告 §8.5 未做②）。本脚本用两条路补上：

  validate 模式：拿一份**现存**的量化图（layer31_cp_q.onnx，与线上同一套 modelslim 配置）
                 与对应 fp16 简化图对比，标定"int8 权重 → 反量化权重"的口径
                 （AscendDequant 的 scale/offset 语义、是否 per-channel 对称），
                 并给出该层的真实重构误差作为基准。
  all 模式：用标定出的同一规则，对 32 层 + 输出头逐个算出**权重 MSE / 相对误差**，
            输出 p3w_weight_mse.json，供与 rollout KL 的排序相关性比对。

用法（quant 环境，CPU）：
  python p3w_weight_mse.py --mode validate
  python p3w_weight_mse.py --mode all
"""

import argparse
import collections
import json
import os
import sys

import numpy as np
import onnx
from onnx import numpy_helper

MODEL_DIR = os.environ.get("RWKV_MODEL_DIR", "/home/disk/models/rwkv7-2.9b")
OUT = os.path.join(MODEL_DIR, "p3w_weight_mse.json")

# 与 build_p1_layers.sh 一致：每层被量化的 6 个权重（attention 4 个 + FFN 2 个）
W_PATTERNS = ("att/receptance", "att/key", "att/value", "att/output",
              "ffn/key", "ffn/value")


def load(path):
    return onnx.load(path, load_external_data=False)


def init_map(model):
    return {t.name: t for t in model.graph.initializer}


def weight_pairs(model):
    """int8 权重按 `<名>_quantized` 存储；返回 [(fp16 名, int8 名)]。"""
    inits = init_map(model)
    pairs = []
    for name, t in inits.items():
        if t.data_type != onnx.TensorProto.INT8:
            continue
        base = name[: -len("_quantized")] if name.endswith("_quantized") else name
        pairs.append((base, name))
    return sorted(pairs)


def validate():
    qpath = os.path.join(MODEL_DIR, "layer31_cp_q.onnx")
    fpath = os.path.join(MODEL_DIR, "layer31_sim.onnx")
    if not (os.path.exists(qpath) and os.path.exists(fpath)):
        print("缺少 layer31_cp_q.onnx / layer31_sim.onnx，无法标定")
        return 1
    qm, fm = load(qpath), load(fpath)
    qi, fi = init_map(qm), init_map(fm)
    print("=== 标定：量化图 vs fp16 图（layer31）===", flush=True)
    report = []
    for base, qname in weight_pairs(qm):
        if base not in fi:
            print("  %-46s fp16 图里无同名权重（跳过）" % base, flush=True)
            continue
        w = numpy_helper.to_array(fi[base]).astype(np.float64)
        q = numpy_helper.to_array(qi[qname]).astype(np.float64)
        transposed = False
        if q.shape != w.shape and q.T.shape == w.shape:
            q, transposed = q.T, True
        if q.shape != w.shape:
            print("  %-46s 形状不匹配 w=%s q=%s" % (base, w.shape, q.shape), flush=True)
            continue
        # 判定真实量化轴：规则 scale = max|w|/127（逐通道），与图中 int8 比对完全一致比例
        best = None
        for axis in (1, 0):
            if w.shape[axis] < 2:
                continue
            red = tuple(0 if d == axis else 1 for d in range(w.ndim))
            amax = np.max(np.abs(w), axis=axis, keepdims=True)
            s = np.where(amax > 0, amax / 127.0, 1.0)
            qhat = np.clip(np.round(w / s), -127, 127)
            agree = float((qhat == q).mean())
            deq = qhat * s
            rel = float(((deq - w) ** 2).sum() / (w ** 2).sum())
            # 图中 int8 用同一规则反解出来的每通道刻度（高信噪比元素上的中位数）
            strong = np.abs(q) >= 64
            if strong.sum() > 0:
                ratio_i = np.where(strong, w / np.where(q == 0, 1, q), np.nan)
                s_est = np.nanmedian(np.where(np.isfinite(ratio_i), ratio_i, np.nan),
                                     axis=axis)
                ratio = float(np.nanmedian(np.asarray(s_est).reshape(-1)
                                           / np.asarray(s).reshape(-1)))
            else:
                ratio = float("nan")
            if best is None or agree > best[3]:
                best = (axis, rel, s_est, agree, ratio)
        axis, rel, _, agree, ratio = best
        row = {"tensor": base, "shape": list(w.shape), "transposed_int8": transposed,
               "channel_axis": axis, "rel_mse": round(rel, 8),
               "int8_exact_match": round(agree, 6),
               "graph_scale_over_max_over127": (round(ratio, 4)
                                                if ratio == ratio else None)}
        report.append(row)
        print("  %-42s %-15s 轴 %d ｜ 相对 MSE %.3e ｜ 与图中 int8 完全一致 %.4f ｜ 图中刻度/(max/127)=%s"
              % (base, str(w.shape), axis, rel, agree,
                 ("%.3f" % ratio) if ratio == ratio else "n/a"), flush=True)
    with open(os.path.join(MODEL_DIR, "p3w_validate.json"), "w") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print("\n标定结果已写出 p3w_validate.json", flush=True)
    return 0


def per_channel_int8(w, n_bits=8):
    """逐行（dim 0）对称 int8：scale = max|w| / 127。

    口径由 validate 模式在 layer31 的量化图上标定：轴 0、scale 恰为 max|w|/127，
    与图中 int8 的完全一致率 99.95%~99.99%（残差来自取整规则）。
    """
    amax = np.max(np.abs(w), axis=0, keepdims=True)
    scale = np.where(amax > 0, amax / 127.0, 1.0)
    q = np.clip(np.round(w / scale), -127, 127)
    return q * scale


def heads_and_weights(path):
    """返回 {张量名: 数组}，只保留被量化的 6 类权重。"""
    m = load(path)
    is_head = os.path.basename(path).startswith("head")
    out = {}
    for t in m.graph.initializer:
        if t.data_type != onnx.TensorProto.FLOAT:
            continue
        arr = numpy_helper.to_array(t)
        if arr.ndim != 2:
            continue
        # 输出头的图里只有那一个大矩阵（65536×2560），按体量筛；层图按 6 类权重名筛
        if (arr.size < 1_000_000) if is_head else \
           (not any(p in t.name for p in W_PATTERNS)):
            continue
        out[t.name] = arr.astype(np.float64)
    if not out:
        print("  （%s 未匹配到权重，初始化器：%s）"
              % (os.path.basename(path),
                 [t.name for t in m.graph.initializer][:12]), flush=True)
    return out


def all_layers():
    result = {}
    for name in ["head"] + ["layer%d" % i for i in range(32)]:
        path = os.path.join(MODEL_DIR, "%s_sim.onnx" % name)
        if not os.path.exists(path):
            print("%-8s 缺 %s_sim.onnx，跳过" % (name, name), flush=True)
            continue
        weights = heads_and_weights(path)
        se = ss = 0.0
        rows = []
        for tname, w in sorted(weights.items()):
            deq = per_channel_int8(w)
            e = float(((deq - w) ** 2).sum())
            s = float((w ** 2).sum())
            se += e
            ss += s
            rows.append({"tensor": tname, "shape": list(w.shape),
                         "mse": round(e / w.size, 10),
                         "rel_mse": round(e / s, 8) if s else None})
        result[name] = {"n_tensors": len(rows), "mse": round(se / max(ss, 1e-30), 8),
                        "abs_mse": round(se / max(sum(np.prod(r["shape"]) for r in rows), 1), 10),
                        "tensors": rows}
        print("%-8s 张量 %d 个 ｜ 相对 MSE %.3e" % (name, len(rows), result[name]["mse"]),
              flush=True)
    with open(OUT, "w") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print("\n已写出 %s（%d 个目标）" % (os.path.basename(OUT), len(result)), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="validate", choices=["validate", "all"])
    args = parser.parse_args()
    return validate() if args.mode == "validate" else all_layers()


if __name__ == "__main__":
    raise SystemExit(main())
