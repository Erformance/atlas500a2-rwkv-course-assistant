#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P3-W 第二步：权重 MSE 作为候选指标的预测力。

协议 §5 把"权重 MSE"列为待比较的候选敏感度之一。它最便宜（不需要任何数据、不需要前向），
所以关键问题是：**它和端到端 rollout KL 的排序是否一致**。

读取：
  p3w_weight_mse.json   逐目标的权重相对 MSE（p3w_weight_mse.py --mode all）
  p3_result.json        逐目标 rollout KL（p3_horizon.py，H=1/8/32/64）
输出：
  p3w_compare.json      与各窗口 KL 的 Spearman、top-8/top-16 重合率
"""

import json
import os

import numpy as np

MODEL_DIR = os.environ.get("RWKV_MODEL_DIR", "/home/disk/models/rwkv7-2.9b")
W = os.path.join(MODEL_DIR, "p3w_weight_mse.json")
P3 = os.path.join(MODEL_DIR, "p3_result.json")
OUT = os.path.join(MODEL_DIR, "p3w_compare.json")


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
    with open(W) as fh:
        w = json.load(fh)
    with open(P3) as fh:
        p3 = json.load(fh)
    table = p3["kl_mean_by_horizon"]

    # 两个文件的键名不同：权重 MSE 用 layer0..layer31/head，rollout KL 用 0..31/head
    def norm(name):
        return name[len("layer"):] if name.startswith("layer") else name

    kl64 = table.get("64", table.get(64, {}))
    names = [n for n in w if norm(n) in kl64]
    print("可用目标 %d 个（权重 MSE 与 rollout KL 都有）" % len(names))
    report = {"n_targets": len(names), "vs": {}}
    for h in sorted(table, key=lambda k: int(k)):
        kl = table[h]
        ns = [n for n in names if norm(n) in kl]
        rho = spearman([w[n]["mse"] for n in ns], [kl[norm(n)] for n in ns])
        order_w = sorted(ns, key=lambda n: w[n]["mse"])
        order_k = sorted(ns, key=lambda n: kl[norm(n)])
        row = {"spearman": round(rho, 4),
               "top8_overlap": round(len(set(order_w[:8]) & set(order_k[:8])) / 8, 3),
               "top16_overlap": round(len(set(order_w[:16]) & set(order_k[:16])) / 16, 3),
               "weight_top8": [norm(n) for n in order_w[:8]],
               "kl_top8": [norm(n) for n in order_k[:8]]}
        report["vs"]["H%s" % h] = row
        print("H=%-3s Spearman %.3f ｜ top8 重合 %.2f ｜ top16 重合 %.2f"
              % (h, rho, row["top8_overlap"], row["top16_overlap"]))

    # 逐层明细（按权重 MSE 排序），便于报告里直接引用
    detail = sorted(((n, round(w[n]["mse"], 8),
                      round(kl64.get(norm(n), float("nan")), 6))
                     for n in names), key=lambda x: x[1])
    print("\n%-8s %-14s %s" % ("目标", "权重相对 MSE", "rollout KL(H=64)"))
    for n, mse, kl in detail[:6] + detail[-4:]:
        print("%-8s %-14.3e %.6f" % (n, mse, kl))
    report["detail"] = [{"target": n, "weight_rel_mse": mse, "kl_h64": kl}
                        for n, mse, kl in detail]

    with open(OUT, "w") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print("\n已写出 %s" % os.path.basename(OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
