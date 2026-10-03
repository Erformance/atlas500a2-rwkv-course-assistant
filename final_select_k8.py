#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""任务书 Task B：**等预算 mixed-precision 选层**（固定 K=8）的计划生成与评估驱动。

思路：计划全部由**已有**的 profiling 产物生成（不重新 profiling），只有"评估计划"需要设备：

| 方法 | 数据来源 | profiling 成本（前向次数） |
| --- | --- | --- |
| Random | 20 个固定 seed，从 32 层里随机取 8 | 0 |
| Weight MSE | `p3w_weight_mse.json` | 0 |
| Local Output Error | `p3_local.json` | 33（每目标 1 次） |
| H=1 / H=8 / H=32 Rollout | `p3_result.json` 的 `kl_mean_by_horizon` | 1 / 8 / 32 per target |
| H=64 / H=128 Reference | `p3_result.json` / `p3d_result.json` | 64 / 128 per target |

用法（设备上）：
  python final_select_k8.py --dry-run        # 只打印会选出哪些层，不写文件、不跑臂
  python final_select_k8.py --write-plans    # 写出 plan_sel_*.json
"""

import argparse
import json
import os
import random

MODEL_DIR = os.environ.get("RWKV_MODEL_DIR", "/home/disk/models/rwkv7-2.9b")
K = 8
N_RANDOM_SEEDS = 20
RANDOM_SEEDS = [1000 + i for i in range(N_RANDOM_SEEDS)]   # 固定并落盘

# Local Output Error 的候选键名（设备侧 p3_local.json 的 schema 在首次运行时打印出来确认）
LOCAL_ERR_KEYS = ("out_err_max", "max_out_err", "output_error", "out_err",
                  "x_err_max", "max_x_err", "x_out_err", "err_max")


def load(name):
    path = os.path.join(MODEL_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def layers_only(names):
    """只保留层（去掉 head），因为所有计划都是"8 层 int8 + 其余 fp16"。"""
    out = []
    for n in names:
        s = str(n)
        if s == "head":
            continue
        out.append(int(s))
    return out


def pick_lowest(table, k=K):
    """table: {target: value} → 取最小的 k 个层号（升序）。"""
    pairs = []
    for name, val in table.items():
        try:
            key = int(name)
        except (TypeError, ValueError):
            continue
        if val is None:
            continue
        pairs.append((float(val), key))
    pairs.sort()
    return sorted(k for _, k in pairs[:k])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write-plans", action="store_true")
    parser.add_argument("--text", default="pilot", help="H=128 参考取自哪份文本的 p3d 结果")
    args = parser.parse_args()

    p3 = load("p3_result.json")
    p3d = load("p3d_result.json")
    wmse = load("p3w_weight_mse.json")
    plocal = load("p3_local.json")

    plans, sources = {}, {}

    # 1) Random（20 seeds）
    for seed in RANDOM_SEEDS:
        rng = random.Random(seed)
        plans["random_seed%d" % seed] = sorted(rng.sample(range(32), K))
        sources["random_seed%d" % seed] = {"method": "random", "seed": seed}

    # 2) Weight MSE（越小越安全）
    if wmse:
        table = {k: (v.get("mse") if isinstance(v, dict) else v) for k, v in wmse.items()}
        plans["weight_mse"] = pick_lowest(table)
        sources["weight_mse"] = {"method": "weight_mse", "source": "p3w_weight_mse.json",
                                 "table": {k: table[k] for k in table}}
    else:
        print("[警告] 缺 p3w_weight_mse.json，跳过 Weight MSE 臂")

    # 3) Local Output Error（越小越安全）
    if plocal:
        sample = plocal.get("layers") or plocal.get("targets") or plocal
        first = next((v for v in sample.values() if isinstance(v, dict)), {})
        keys = [k for k in LOCAL_ERR_KEYS if k in first]
        print("[info] p3_local.json 里可用键：%s" % sorted(first.keys()))
        if keys:
            key = keys[0]
            table = {n: (v.get(key) if isinstance(v, dict) else None) for n, v in sample.items()}
            plans["local_output_error"] = pick_lowest(table)
            sources["local_output_error"] = {"method": "local_output_error",
                                             "metric_key": key, "source": "p3_local.json"}
        else:
            print("[警告] p3_local.json 里没有已知的误差键，跳过 Local Output Error 臂")
    else:
        print("[警告] 缺 p3_local.json，跳过 Local Output Error 臂")

    # 4) Rollout H=1/8/32/64 与 H=128 参考
    if p3:
        table = p3.get("kl_mean_by_horizon", {})
        for h in ("1", "8", "32", "64"):
            if h in table:
                plans["h%s_rollout" % h] = pick_lowest(table[h])
                sources["h%s_rollout" % h] = {"method": "rollout", "H": int(h),
                                              "source": "p3_result.json"}
    if p3d:
        t = p3d.get("texts", {}).get(args.text, {})
        table = t.get("kl_mean_by_horizon", {})
        h128 = table.get("128") or table.get(128)
        if h128:
            plans["h128_reference"] = pick_lowest(h128)
            sources["h128_reference"] = {"method": "rollout", "H": 128,
                                         "source": "p3d_result.json(%s)" % args.text}
    p1plan = load("plan_p1_top8.json")
    if p1plan:
        plans["p1_plan_top8"] = sorted(int(x) for x in p1plan.get("layers", []))
        sources["p1_plan_top8"] = {"method": "existing_plan", "source": "plan_p1_top8.json"}

    print("\n=== 生成的计划（K=%d）===" % K)
    for name, layers in plans.items():
        print("  %-18s %s" % (name, layers))
    if not args.dry_run:
        for name, layers in plans.items():
            # 所有计划都用同一套校准（线上 _p1_q）：这样"选层方法"是唯一变量。
            # P3 阶段已把 33 个目标全部编译为 _p1_q.om，因此本任务零编译成本。
            plan = {"suffix": "_p1_q", "layers": layers, "head": False}
            if not args.write_plans:
                continue
            with open(os.path.join(MODEL_DIR, "plan_sel_%s.json" % name), "w",
                      encoding="utf-8") as fh:
                json.dump(plan, fh, ensure_ascii=False, indent=2)
        if args.write_plans:
            out = {"task": "B. Equal-budget mixed-precision selection (K=8)",
                   "k": K, "random_seeds": RANDOM_SEEDS,
                   "plans": plans, "sources": sources,
                   "profiling_cost_forward_passes": {
                       "random": 0, "weight_mse": 0,
                       "local_output_error": 33,
                       "h1_rollout": 33, "h8_rollout": 33 * 8,
                       "h32_rollout": 33 * 32, "h64_rollout": 33 * 64,
                       "h128_reference": 33 * 128}}
            with open(os.path.join(MODEL_DIR, "final_selection_k8_plans.json"), "w",
                      encoding="utf-8") as fh:
                json.dump(out, fh, ensure_ascii=False, indent=2)
            print("\n计划文件已写出：plan_sel_*.json、final_selection_k8_plans.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
