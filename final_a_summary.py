#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""任务书 Task A 的汇总：把四条校准臂 × 两份文本 × 两个窗口的成绩整理成
`final_calibration_baselines.json`（任务书 §8 要求的交付物）。

输入（设备侧产物）：
  p2b_b0_manifest.json          A1（32 段 × 年龄 0）
  p1_a2_capture_manifest.json   A2（16 段 × 2 个随机位置）
  p2b_b1_manifest.json          A3（16 段 × {0,256}）
  p1_capture_manifest.json      A4（32 段 × {0,64,256}，高预算参照）
  final_a_compare_<text>_<64|128>.json   四臂 vs fp16 的 KL / top-1 / wkv 偏差 / 速度

判定规则按任务书 §2.5，用相对差 10% 作为"≈"的阈值并在输出里写明。
"""

import argparse
import json
import os

MODEL_DIR = os.environ.get("RWKV_MODEL_DIR", "/home/disk/models/rwkv7-2.9b")
OUT = os.path.join(MODEL_DIR, "final_calibration_baselines.json")
TEXTS = ("pilot", "p1test")
HORIZONS = (64, 128)
ARM_INFO = {
    "a1": ("A1_beginning_only", "_b0_q", "p2b_b0_manifest.json"),
    "a2": ("A2_random_trajectory", "_a2_q", "p1_a2_capture_manifest.json"),
    "a3": ("A3_age_stratified", "_b1_q", "p2b_b1_manifest.json"),
    "a4": ("A4_dense_age_stratified", "_p1_q", "p1_capture_manifest.json"),
}


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--t0", type=int, default=0)
    parser.add_argument("--t-cap", type=int, default=0)
    parser.add_argument("--t-quant", type=int, default=0)
    parser.add_argument("--t-eval", type=int, default=0)
    args = parser.parse_args()

    base_manifest = load_json(os.path.join(MODEL_DIR, "manifest.json")) or {}

    arms = {}
    for key, (label, suffix, cman) in ARM_INFO.items():
        cm = load_json(os.path.join(MODEL_DIR, cman)) or {}
        arms[key] = {
            "label": label,
            "om_suffix": suffix,
            "n_samples": cm.get("n_samples"),
            "n_segments": cm.get("n_segments"),
            "state_age_ages": cm.get("ages") or cm.get("state_age_distribution"),
            "split_sha1": cm.get("split_sha1"),
            "seed": cm.get("seed"),
            "capture_manifest": cman,
            "equal_budget_group": key in ("a1", "a2", "a3"),
        }

    metrics = {}
    for text in TEXTS:
        metrics[text] = {}
        for h in HORIZONS:
            rep = load_json(os.path.join(MODEL_DIR, "final_a_compare_%s_%d.json" % (text, h)))
            if not rep:
                continue
            row = {}
            for key in ARM_INFO:
                entry = rep.get("arms", {}).get(ARM_INFO[key][1], {})
                row[ARM_INFO[key][0]] = {
                    "kl_mean": entry.get("kl_mean"),
                    "top1_agreement": entry.get("top1_agreement"),
                    "wkv_rel_mean": entry.get("wkv_rel_mean"),
                    "decode_tok_s": entry.get("decode_tok_s"),
                    "n_quant_layers": entry.get("n_quant_layers"),
                }
            metrics[text]["H%d" % h] = {"tokens": rep.get("tokens"), "arms": row}

    # 判定：用"最贴近部署"的那一档（128 步）在两份文本上的 KL 均值做比较
    verdict = {}
    for text in TEXTS:
        row = metrics.get(text, {}).get("H128", {}).get("arms")
        if not row:
            continue
        kl = {k: row[ARM_INFO[k][0]].get("kl_mean") for k in ("a1", "a2", "a3")}
        if any(v is None for v in kl.values()):
            continue
        def close(x, y):
            hi = max(abs(x), abs(y)) + 1e-12
            return abs(x - y) / hi <= 0.10
        if kl["a3"] < kl["a2"] and not close(kl["a2"], kl["a3"]):
            tag = "age_stratified_better_than_random"
            claim = ("固定 calibration budget 下，state-age-aware stratification 改善 recurrent LM PTQ"
                     "（A3 < A2，且 A2/A3 差异超过 10%）")
        elif close(kl["a2"], kl["a3"]):
            tag = "random_equals_stratified"
            claim = ("关键在覆盖成熟 recurrent state，而不是 state-age stratification 本身"
                     "（A2 ≈ A3，差异在 10% 以内）")
        else:
            tag = "random_better_than_stratified"
            claim = ("状态覆盖关键，但人工分层不优于随机真实轨迹（A2 < A3）——如实保留负结果")
        verdict[text] = {
            "kl_H128": kl,
            "a1_vs_a2_ratio": round(kl["a1"] / max(kl["a2"], 1e-12), 3),
            "a3_vs_a2_ratio": round(kl["a3"] / max(kl["a2"], 1e-12), 3),
            "tag": tag, "paper_claim": claim,
        }

    report = {
        "task": "A. Calibration fair baseline（任务书 §2）",
        "date": "2026-10-03",
        "seed": args.seed,
        "fixed_conditions": {
            "model": "RWKV-7 G1j 2.9B",
            "quant_toolchain": "modelslim ONNX post-training quant（per-channel int8, method=1）",
            "int8_layer_plan": [24, 6, 11, 22, 23, 20, 25, 28],
            "head": False,
            "note": "A1/A2/A3 样本数完全一致（各 32）；A4（96）只作高预算参照，不参与等成本比较",
        },
        "arms": arms,
        "metrics": metrics,
        "verdict": verdict,
        "timing_seconds": {
            "capture": args.t_cap - args.t0 if args.t_cap and args.t0 else None,
            "quantize_and_atc": args.t_quant - args.t_cap if args.t_quant and args.t_cap else None,
            "evaluation": args.t_eval - args.t_quant if args.t_eval and args.t_quant else None,
            "total": args.t_eval - args.t0 if args.t_eval and args.t0 else None,
        },
        "hashes": {
            "model_revision": base_manifest.get("model_revision"),
            "tokenizer_hash": base_manifest.get("tokenizer_hash"),
            "data_split_sha1_calib": arms["a4"].get("split_sha1"),
            "note": "OM 逐文件 sha256 见设备 manifest.json 的 candidate_plan 段",
        },
        "reproduce": [
            "bash a2_chain.sh                     # 采 A2 → 打包 → 编译 → 评估 → 汇总",
            "python capture_calib_a2.py --limit 16 --per-seg 2 --seed 20261003",
            "python ab_run_arm.py --plan-json \"$(cat plan_a2_top8.json)\" --tokens 128 --text pilot --out arm_a_a2_pilot_128.npz",
        ],
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print("已写出 %s" % os.path.basename(OUT))
    for text, v in verdict.items():
        print("  [%s] A1 %.4f / A2 %.4f / A3 %.4f → %s"
              % (text, v["kl_H128"]["a1"], v["kl_H128"]["a2"], v["kl_H128"]["a3"], v["tag"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
