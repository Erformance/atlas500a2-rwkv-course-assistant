#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P3 方向指标第三步：把 γᵢ（输出加权方向灵敏度）与已有指标/rollout KL 比预测力。

比较对象（全部来自同一批 32 层）：
  * γᵢ = |gᵢ·δᵢ|                  ← p3b_dir.json（方向指标，本次新增）
  * ‖δᵢ‖                          ← p3b_dir.json（纯大小，等价于局部误差量级）
  * 局部单步输出相对误差          ← p3_local.json（已有）
  * 64 步 rollout KL              ← p3_result.json（真值排序）

输出 Spearman、top-8/top-16 重合率，并给出"按 γ 排序选前 8 层"的计划文件
（层都已编译，零编译成本，可直接用 ab_run_arm.py 实测计划 KL）。

用法: python p3b_compare.py
"""

import json
import os

MODEL_DIR = os.environ.get("RWKV_MODEL_DIR", "/home/disk/models/rwkv7-2.9b")


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


def load(name):
    path = os.path.join(MODEL_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main():
    dirj = load("p3_dir.json") or load("p3b_dir.json")
    localj = load("p3_local.json")
    rollj = load("p3_result.json")
    if not (dirj and rollj):
        raise SystemExit("缺少 p3b_dir.json 或 p3_result.json")
    kl64 = rollj["kl_mean_by_horizon"]["64"]

    layers = [n for n in kl64 if n != "head"]
    metrics = {}
    for i, r in dirj["layers"].items():
        if i in layers and r.get("dir_score") is not None:
            metrics.setdefault("dir_score", {})[i] = r["dir_score"]
            if r.get("delta_norm") is not None:
                metrics.setdefault("delta_norm", {})[i] = r["delta_norm"]
    if localj:
        for i, r in localj["layers"].items():
            if i in layers:
                metrics.setdefault("rel_err_max", {})[i] = r["rel_err_max"]

    order_kl = sorted(layers, key=lambda n: kl64[n])
    report = {"n_layers": len(layers), "metrics": {}}
    print("%-14s %-12s %-10s %-10s" % ("指标", "Spearman", "top8 重合", "top16 重合"))
    for name, vals in metrics.items():
        shared = [n for n in layers if n in vals]
        if len(shared) < 5:
            continue
        rho = spearman([vals[n] for n in shared], [kl64[n] for n in shared])
        order_m = sorted(shared, key=lambda n: vals[n])
        ov = {k: len(set(order_m[:k]) & set(order_kl[:k])) / k for k in (8, 16)}
        report["metrics"][name] = {"spearman": round(rho, 4), "top8": ov[8],
                                   "top16": ov[16], "n": len(shared)}
        print("%-14s %-12.3f %-10.2f %-10.2f" % (name, rho, ov[8], ov[16]))

    # 按 γ 排序的前 8 层 → 生成计划文件（零编译成本，可直接实测）
    if "dir_score" in metrics:
        order_dir = sorted(metrics["dir_score"], key=lambda n: metrics["dir_score"][n])
        plan = {"suffix": "_p1_q", "layers": [int(x) for x in order_dir[:8]], "head": False}
        path = os.path.join(MODEL_DIR, "plan_dir_top8.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(plan, fh, ensure_ascii=False, indent=2)
        report["plan_from_dir_top8"] = plan
        print("\n按方向指标选出的前 8 层：%s" % plan["layers"])
        print("（已写 %s；可用 ab_run_arm.py 实测计划 KL，与 top8/top16 对照）" % path)

    with open(os.path.join(MODEL_DIR, "p3b_compare.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print("已写出 p3b_compare.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
