#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""任务书 Task C：把"差 2–3 题"升级为可发表的统计结论。

输入：两份逐题 JSONL（fp16 与均衡档），由
`p4_run_eval.py --samples-out <file>.jsonl` 产出（`results["samples"]` 落盘）。

输出：`benchmark_paired_stats.json`，含
  * ΔAcc = Acc(INT8) − Acc(FP16)：paired bootstrap（默认 5000 次）的 mean 与 95% CI；
  * McNemar：discordant pair 计数（FP16 对/INT8 错 = b，FP16 错/INT8 对 = c）与精确二项 p 值；
  * LAMBADA：若逐题里带 NLL 类字段，则给 ΔNLL 的 paired bootstrap（mean/median/95% CI）与 PPL 变化；
  * 结论措辞：差异不显著时写"当前样本量下未观察到统计上明确的 accuracy difference"。

用法：
  python benchmark_paired_stats.py --fp16 samples_fp16_piqa.jsonl \
      --int8 samples_balanced_piqa.jsonl --task piqa --out-csv /dev/stdout
"""

import argparse
import json
import math
import os
import random


def load_items(path):
    items = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            doc = r.get("doc_id", len(items))
            items[doc] = r
    return items


def first(x):
    if isinstance(x, (list, tuple)):
        return x[0] if x else None
    return x


def correctness(row, metric):
    v = row.get(metric)
    v = first(v)
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    try:
        return float(v) >= 0.5
    except (TypeError, ValueError):
        return None


def paired_bootstrap(a, b, iters, seed=20261003, stat="mean"):
    """a、b 为等长的逐题数值（或 bool→0/1）。返回 (mean_diff, lo, hi, median_diff)。"""
    rng = random.Random(seed)
    n = len(a)
    diffs = [float(x) - float(y) for x, y in zip(a, b)]
    mean_diff = sum(diffs) / n
    srt = sorted(diffs)
    median = srt[n // 2] if n % 2 else 0.5 * (srt[n // 2 - 1] + srt[n // 2])
    boots = []
    for _ in range(iters):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        boots.append(s / n)
    boots.sort()
    lo = boots[int(0.025 * iters)]
    hi = boots[int(0.975 * iters) - 1]
    return mean_diff, lo, hi, median


def mcnemar_pvalue(b, c):
    """精确二项检验（p=0.5），不需要 scipy。b、c 为两个 discordant 计数。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp16", required=True)
    parser.add_argument("--int8", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--metric", default="", help="默认按 acc→acc_norm→exact_match 依次尝试")
    parser.add_argument("--iters", type=int, default=5000)
    parser.add_argument("--nll-field", default="",
                        help="逐题 NLL 字段名；不填则自动探测（nll/ll/loglikelihood/seq_loglikelihood）")
    parser.add_argument("--out", default="benchmark_paired_stats.json")
    args = parser.parse_args()

    a, b = load_items(args.fp16), load_items(args.int8)
    common = sorted(set(a) & set(b), key=lambda x: (isinstance(x, str), x))
    if not common:
        raise SystemExit("两份逐题文件没有共同的 doc_id，检查是否能对齐")

    metrics = [args.metric] if args.metric else ["acc", "acc_norm", "exact_match"]
    report = {"task": args.task, "n_items": len(common),
              "iters": args.iters, "metrics": {}, "mcnemar": {}, "nll": None}

    for m in metrics:
        pa = [correctness(a[d], m) for d in common]
        pb = [correctness(b[d], m) for d in common]
        if any(x is None for x in pa) or any(x is None for x in pb):
            continue
        acc_a = sum(pa) / len(pa)
        acc_b = sum(pb) / len(pb)
        mean_d, lo, hi, med = paired_bootstrap(pb, pa, args.iters)
        b_cnt = sum(1 for x, y in zip(pa, pb) if x and not y)   # fp16 对 / int8 错
        c_cnt = sum(1 for x, y in zip(pa, pb) if y and not x)   # fp16 错 / int8 对
        p = mcnemar_pvalue(b_cnt, c_cnt)
        report["metrics"][m] = {
            "acc_fp16": round(acc_a, 4), "acc_int8": round(acc_b, 4),
            "delta_acc_mean": round(mean_d, 4),
            "delta_acc_ci95": [round(lo, 4), round(hi, 4)],
            "delta_acc_median": round(med, 4),
            "significant": bool(lo > 0 or hi < 0),
            "wording": ("当前样本量下未观察到统计上明确的 accuracy difference"
                        if (lo <= 0 <= hi) else
                        "差异的 95% CI 不跨 0，需按方向解释"),
        }
        report["mcnemar"][m] = {"fp16_correct_int8_wrong": b_cnt,
                                "fp16_wrong_int8_correct": c_cnt,
                                "discordant_total": b_cnt + c_cnt,
                                "p_value": round(p, 4)}

    # NLL / PPL（LAMBADA 这类无法用 accuracy 表达的指标）
    fields = [args.nll_field] if args.nll_field else \
             ["nll", "ll", "loglikelihood", "seq_loglikelihood"]
    for f in fields:
        va = [a[d].get(f) for d in common]
        vb = [b[d].get(f) for d in common]
        if any(v is None for v in va) or any(v is None for v in vb):
            continue
        va = [float(first(v)) for v in va]
        vb = [float(first(v)) for v in vb]
        mean_d, lo, hi, med = paired_bootstrap(vb, va, args.iters)
        ppl_a = math.exp(-sum(va) / len(va))
        ppl_b = math.exp(-sum(vb) / len(vb))
        report["nll"] = {
            "field": f,
            "nll_fp16_mean": round(sum(va) / len(va), 4),
            "nll_int8_mean": round(sum(vb) / len(vb), 4),
            "delta_nll_mean": round(mean_d, 4),
            "delta_nll_median": round(med, 4),
            "delta_nll_ci95": [round(lo, 4), round(hi, 4)],
            "ppl_fp16": round(ppl_a, 4), "ppl_int8": round(ppl_b, 4),
            "ppl_change_pct": round((ppl_b / ppl_a - 1) * 100, 2),
        }
        break

    out_path = args.out
    if os.path.dirname(out_path):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    for m, v in report["metrics"].items():
        mc = report["mcnemar"][m]
        print("%-14s acc %.4f → %.4f ｜ Δ %+.4f（95%% CI %+.4f ~ %+.4f）｜ McNemar b=%d c=%d p=%.4f"
              % (m, v["acc_fp16"], v["acc_int8"], v["delta_acc_mean"],
                 v["delta_acc_ci95"][0], v["delta_acc_ci95"][1],
                 mc["fp16_correct_int8_wrong"], mc["fp16_wrong_int8_correct"], mc["p_value"]))
    if report["nll"]:
        n = report["nll"]
        print("NLL(%s) %.4f → %.4f ｜ Δ %+.4f（95%% CI %+.4f ~ %+.4f）｜ PPL %.3f → %.3f（%+.2f%%）"
              % (n["field"], n["nll_fp16_mean"], n["nll_int8_mean"], n["delta_nll_mean"],
                 n["delta_nll_ci95"][0], n["delta_nll_ci95"][1],
                 n["ppl_fp16"], n["ppl_int8"], n["ppl_change_pct"]))
    print("已写出 %s" % out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
