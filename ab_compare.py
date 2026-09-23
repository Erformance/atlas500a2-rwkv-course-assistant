#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P2：比较各臂与 fp16 的连续轨迹指标（在同一台机器上、同一段 token 序列）。

用法: python ab_compare.py --ref arm_fp16.npz --arm arm_q.npz --arm arm_cp.npz
输出: ab_logits.json（含 KL、top-1 一致率、速度、状态摘要）
"""

import argparse
import json
import os
import sys

import numpy as np

MODEL_DIR = "/home/disk/models/rwkv7-2.9b"


def softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", required=True)
    parser.add_argument("--arm", action="append", required=True)
    parser.add_argument("--meta", action="append", default=[])
    parser.add_argument("--out", default=os.path.join(MODEL_DIR, "ab_logits.json"))
    args = parser.parse_args()

    metas = {}
    for p in args.meta:
        with open(p) as fh:
            m = json.load(fh)
        metas[m["suffix"]] = m

    ref = np.load(args.ref)
    ref_logits = ref["logits"]
    ref_top1 = ref_logits.argmax(axis=1)
    report = {"tokens": int(ref_logits.shape[0]),
              "reference": metas.get("", {}), "arms": {}}

    for path in args.arm:
        data = np.load(path)
        logits = data["logits"]
        suffix = os.path.basename(path).replace("arm_", "").replace(".npz", "")
        key = "_" + suffix if not suffix.startswith("_") else suffix
        kls, agree = [], []
        for t in range(logits.shape[0]):
            p = softmax(ref_logits[t])
            q = softmax(logits[t])
            kls.append(float(np.sum(p * (np.log(p + 1e-12) - np.log(q + 1e-12)))))
            agree.append(bool(logits[t].argmax() == ref_top1[t]))
        meta = metas.get(key, {})
        entry = {
            "tokens": int(logits.shape[0]),
            "kl_mean": round(float(np.mean(kls)), 4),
            "kl_max": round(float(np.max(kls)), 4),
            "kl_first10_mean": round(float(np.mean(kls[:10])), 4),
            "kl_last10_mean": round(float(np.mean(kls[-10:])), 4),
            "top1_agreement": round(float(np.mean(agree)), 4),
            "wkv_ref_vs_arm": None,
        }
        if "wkv" in data and "wkv" in ref:
            a, b = ref["wkv"], data["wkv"]
            entry["wkv_rel_mean"] = round(float(np.mean(np.abs(a - b) / (np.abs(a) + 1e-9))), 4)
        entry.update({k: v for k, v in meta.items()
                      if k in ("n_quant_layers", "n_fp16_layers", "prefill_s",
                               "decode_ms_per_token", "decode_tok_s")})
        report["arms"][key] = entry
        print("臂 %-8s int8 %2d 层 ｜ %6.2f tok/s ｜ KL 均值 %7.3f（前10 %7.3f / 后10 %7.3f）"
              "｜ top-1 一致 %5.1f%% ｜ wkv 相对偏差 %s"
              % (key, entry.get("n_quant_layers", -1), entry.get("decode_tok_s", 0),
                 entry["kl_mean"], entry["kl_first10_mean"], entry["kl_last10_mean"],
                 entry["top1_agreement"] * 100, entry.get("wkv_rel_mean")))

    with open(args.out, "w") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print("\n已写出 %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
