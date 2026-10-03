#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""任务书 Task B 的汇总：把各选层策略的实测成绩整理成任务书 §3.5 的表格
（`final_selection_k8.json`）。

口径：
  * Sequence KL  = 128 步逐 token KL 的**均值**；
  * 128-step KL  = 同一条轨迹的 KL **总和**（整条轨迹的累计偏离）；
  * Top-1        = 与 fp16 参考 argmax 一致的比例；
  * tok/s        = 纯解码口径（来自 ab_run_arm 的 meta）；
  * RSS          = 与"int8 层数 K"有关、与"选哪 8 层"基本无关（每层 .om 体积近似），
                   故各计划共用 P5 实测的均衡档数值，并在输出里标注来源。
"""

import argparse
import json
import os

import numpy as np

MODEL_DIR = os.environ.get("RWKV_MODEL_DIR", "/home/disk/models/rwkv7-2.9b")
OUT = os.path.join(MODEL_DIR, "final_selection_k8.json")


def softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def kl_per_step(ref, logits):
    out = []
    for t in range(logits.shape[0]):
        p, q = softmax(ref[t]), softmax(logits[t])
        out.append(float(np.sum(p * (np.log(p + 1e-12) - np.log(q + 1e-12)))))
    return np.array(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--texts", default="pilot p1test")
    parser.add_argument("--rss-mb", type=float, default=3070.0,
                        help="8 层 int8 的进程 RSS（默认取自 P5 均衡档实测）")
    args = parser.parse_args()

    plans_file = os.path.join(MODEL_DIR, "final_selection_k8_plans.json")
    with open(plans_file, encoding="utf-8") as fh:
        meta = json.load(fh)
    plans, sources = meta["plans"], meta["sources"]
    costs = meta["profiling_cost_forward_passes"]

    def cost_of(name):
        if name.startswith("random"):
            return costs["random"]
        if name.startswith("weight_mse"):
            return costs["weight_mse"]
        if name.startswith("local_output_error"):
            return costs["local_output_error"]
        for h in (1, 8, 32, 64, 128):
            if name.startswith("h%d_rollout" % h):
                return costs.get("h%d_rollout" % h, 33 * h)
        if name.startswith("h128_reference"):
            return costs["h128_reference"]
        if name.startswith("p1_plan_top8"):
            return None
        return None

    rows = {}
    for text in args.texts.split():
        ref_path = os.path.join(MODEL_DIR, "arm_fp16_128_%s.npz" % text)
        if not os.path.exists(ref_path):
            print("[警告] 缺参考臂 %s，跳过 %s" % (ref_path, text))
            continue
        ref = np.load(ref_path)["logits"]
        ref_top1 = ref.argmax(axis=1)
        per_text = {}
        for name in plans:
            arm = os.path.join(MODEL_DIR, "arm_sel_%s_%s_128.npz" % (name, text))
            if not os.path.exists(arm):
                continue
            data = np.load(arm)
            logits = data["logits"]
            kls = kl_per_step(ref, logits)
            agree = float(np.mean(logits.argmax(axis=1) == ref_top1))
            row = {
                "sequence_kl_mean": round(float(kls.mean()), 4),
                "sequence_kl_max": round(float(kls.max()), 4),
                "trajectory_kl_sum_128": round(float(kls.sum()), 3),
                "top1_agreement": round(agree, 4),
                "layers": plans[name],
            }
            if "wkv" in data and "wkv" in np.load(ref_path):
                rw = np.load(ref_path)["wkv"]
                row["wkv_rel_mean"] = round(
                    float(np.mean(np.abs(rw - data["wkv"]) / (np.abs(rw) + 1e-9))), 4)
            meta_path = os.path.join(MODEL_DIR, "meta_sel_%s_%s_128.json" % (name, text))
            if os.path.exists(meta_path):
                with open(meta_path) as fh:
                    m = json.load(fh)
                row["decode_tok_s"] = m.get("decode_tok_s")
                row["n_quant_layers"] = m.get("n_quant_layers")
            row["profiling_cost"] = cost_of(name)
            row["profiling_cost_unit"] = "forward passes（33 目标合计）"
            row["source"] = sources.get(name, {})
            per_text[name] = row
        rows[text] = per_text

    # 随机臂的统计量（mean±std、best/median/worst）
    random_stats = {}
    for text, per_text in rows.items():
        kls = [(n, v["sequence_kl_mean"], v["top1_agreement"])
               for n, v in per_text.items() if n.startswith("random_seed")]
        if not kls:
            continue
        vals = np.array([k for _, k, _ in kls])
        tops = np.array([t for _, _, t in kls])
        order = np.argsort(vals)
        random_stats[text] = {
            "n_seeds": len(kls),
            "kl_mean": round(float(vals.mean()), 4),
            "kl_std": round(float(vals.std(ddof=1)), 4),
            "kl_best": round(float(vals[order[0]]), 4),
            "kl_median": round(float(np.median(vals)), 4),
            "kl_worst": round(float(vals[order[-1]]), 4),
            "top1_mean": round(float(tops.mean()), 4),
            "top1_std": round(float(tops.std(ddof=1)), 4),
            "best_seed": kls[order[0]][0], "worst_seed": kls[order[-1]][0],
        }

    # 任务书 §3.5 的表格（以 pilot 为主表，p1test 并列给出）
    table = []
    for method, name in (("Random", None), ("Weight MSE", "weight_mse"),
                         ("Local Error", "local_output_error"),
                         ("H=1", "h1_rollout"), ("H=8", "h8_rollout"),
                         ("H=32", "h32_rollout"), ("H=64 (ref.)", "h64_rollout"),
                         ("H=128 (ref.)", "h128_reference"),
                         ("线上 P1 计划", "p1_plan_top8")):
        if name is None:
            st = random_stats.get("pilot") or random_stats.get("p1test")
            if not st:
                continue
            table.append({"selection_method": method, "K": 8,
                          "profiling_cost": 0,
                          "sequence_kl": "%s ± %s" % (st["kl_mean"], st["kl_std"]),
                          "trajectory_kl_128": None,
                          "top1": "%s ± %s" % (st["top1_mean"], st["top1_std"]),
                          "tok_s": None,
                          "note": "20 seeds：best %s / median %s / worst %s"
                                  % (st["kl_best"], st["kl_median"], st["kl_worst"])})
            continue
        row = rows.get("pilot", {}).get(name)
        if not row:
            continue
        table.append({"selection_method": method, "K": 8,
                      "profiling_cost": row["profiling_cost"],
                      "sequence_kl": row["sequence_kl_mean"],
                      "trajectory_kl_128": row["trajectory_kl_sum_128"],
                      "top1": row["top1_agreement"],
                      "tok_s": row.get("decode_tok_s"),
                      "layers": row["layers"]})

    report = {
        "task": "B. Equal-budget mixed-precision selection（任务书 §3，K=8）",
        "date": "2026-10-03",
        "k": 8,
        "random_seeds": meta["random_seeds"],
        "reference": "fp16（arm_fp16_128_<text>.npz），全部计划共用同一套 P1 校准（_p1_q）",
        "table_pilot": table,
        "per_plan": rows,
        "random_stats": random_stats,
        "memory": {"rss_mb": args.rss_mb,
                   "source": "P5 报告 §2（同为 8 层 int8 的均衡档）",
                   "note": "RSS 由 int8 层数 K 决定，与具体选哪 8 层基本无关"},
        "profiling_cost_definition": "前向次数（33 个目标合计）；random/weight_mse 为 0",
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print("已写出 %s" % os.path.basename(OUT))
    for t in table:
        print("  %-14s cost=%-6s KL=%-18s 128步KL=%-10s top1=%-16s tok/s=%s"
              % (t["selection_method"], t["profiling_cost"], t["sequence_kl"],
                 t["trajectory_kl_128"], t["top1"], t["tok_s"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
