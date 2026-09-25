#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P3 汇总：把"局部指标"（单步输出/状态相对误差）与"rollout KL"（各 H）对齐比较。

协议 §5 要比较不同敏感度指标对"完整短 rollout KL"的预测力。这里用同一批 32 层：
  * 局部指标：p3_local.json 里同一份真实校准输入下的输出相对误差（x / wkv / 最大）
  * rollout 指标：p3_result.json 里每个目标的 KL（H=1/8/32/64）
算 Spearman 排序相关与 top-8 重合率，输出 p3_combined.json。

用法: python p3_combine.py
"""

import json
import os

MODEL_DIR = "/home/disk/models/rwkv7-2.9b"


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def main():
    with open(os.path.join(MODEL_DIR, "p3_result.json"), encoding="utf-8") as fh:
        horizon = json.load(fh)
    with open(os.path.join(MODEL_DIR, "p3_local.json"), encoding="utf-8") as fh:
        local = json.load(fh)["layers"]

    kl64 = horizon["kl_mean_by_horizon"]["64"]
    kl8 = horizon["kl_mean_by_horizon"]["8"]
    kl1 = horizon["kl_mean_by_horizon"]["1"]

    layers = [n for n in kl64 if n != "head" and n in local]
    report = {"layers": layers, "metrics": {}}

    def order_of(vals):
        return sorted(vals, key=lambda n: vals[n])

    order_kl = order_of({n: kl64[n] for n in layers})
    for metric, fn in (("rel_err_max", lambda d: d["rel_err_max"]),
                       ("rel_err_x", lambda d: d["rel_err"].get("x")),
                       ("rel_err_wkv", lambda d: d["rel_err"].get("wkv_out_0")),
                       ("rel_err_ffn", lambda d: d["rel_err"].get("ffn_shift_out_0"))):
        vals = {}
        for n in layers:
            v = fn(local[n])
            if v is not None:
                vals[n] = v
        if len(vals) < 5:
            continue
        shared = [n for n in layers if n in vals]
        rho = spearman([vals[n] for n in shared], [kl64[n] for n in shared])
        order_m = order_of(vals)
        overlap = {("top%d" % k): len(set(order_m[:k]) & set(order_kl[:k])) / k
                   for k in (8, 16)}
        report["metrics"][metric] = {"spearman_vs_kl64": round(rho, 4),
                                     "topk_overlap": overlap,
                                     "n": len(shared)}
        print("%-12s 与 H=64 rollout KL 的 Spearman %6.3f｜top8 重合 %.2f｜top16 重合 %.2f"
              % (metric, rho, overlap["top8"], overlap["top16"]))

    # 局部指标之间的一致性（例：输出误差最大的是不是状态误差最大的）
    # 局部指标之间的一致性：x 输出误差、状态（wkv）误差、FFN 状态误差两两比
    def col(key):
        return [local[n]["rel_err"].get(key, 0.0) for n in layers]

    pairs = {"x_vs_wkv": ("x", "wkv_out_0"),
             "x_vs_ffn": ("x", "ffn_shift_out_0"),
             "wkv_vs_ffn": ("wkv_out_0", "ffn_shift_out_0")}
    report["local_self_consistency"] = {}
    for tag, (a, b) in pairs.items():
        xs, ys = col(a), col(b)
        if len(set(xs)) < 2 or len(set(ys)) < 2:
            report["local_self_consistency"][tag] = None
            continue
        r = spearman(xs, ys)
        report["local_self_consistency"][tag] = round(r, 4)
        print("局部指标自洽性 %-12s Spearman %6.3f" % (tag, r))

    with open(os.path.join(MODEL_DIR, "p3_combined.json"), "w") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print("已写出 p3_combined.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
