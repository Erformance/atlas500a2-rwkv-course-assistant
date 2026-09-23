#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""逐层敏感度测量：每次只把一层（或输出头）换成 int8，测端到端 KL / top-1 一致率。

目的：找出"压了没事"的层（白送速度）与"压了就崩"的层（必须保 fp16），
为混合精度计划提供依据。每层用独立子进程运行，避免设备内存累积。

用法: python sensitivity.py [--suffix _cp_q] [--tokens 64] [--targets all]
前置: arm_fp16.npz（fp16 参考的 logits）
输出: sensitivity.json
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

MODEL_DIR = "/home/disk/models/rwkv7-2.9b"
PY = "/home/disk/miniconda3/envs/npu22/bin/python"


def softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def kl_top1(ref_logits, logits):
    kls, agree = [], []
    for t in range(logits.shape[0]):
        p, q = softmax(ref_logits[t]), softmax(logits[t])
        kls.append(float(np.sum(p * (np.log(p + 1e-12) - np.log(q + 1e-12)))))
        agree.append(bool(logits[t].argmax() == ref_logits[t].argmax()))
    return float(np.mean(kls)), float(np.mean(agree))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", default="_cp_q")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--targets", default="all")
    parser.add_argument("--ref", default=os.path.join(MODEL_DIR, "arm_fp16.npz"))
    parser.add_argument("--out", default=os.path.join(MODEL_DIR, "sensitivity.json"))
    args = parser.parse_args()

    ref = np.load(args.ref)["logits"]
    if args.targets == "all":
        targets = ["head"] + [str(i) for i in range(32)]
    else:
        targets = args.targets.split(",")

    results = {}
    if os.path.exists(args.out):                      # 支持中断续跑
        with open(args.out) as fh:
            results = json.load(fh).get("targets", {})

    for t in targets:
        if t in results:
            print("跳过 %s（已有）" % t)
            continue
        if t == "head":
            plan = {"suffix": args.suffix, "layers": [], "head": True}
        else:
            plan = {"suffix": args.suffix, "layers": [int(t)], "head": False}
        tmp = "/tmp/arm_sens_%s.npz" % t
        cmd = [PY, os.path.join(MODEL_DIR, "ab_run_arm.py"),
               "--plan-json", json.dumps(plan), "--tokens", str(args.tokens),
               "--out", tmp]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if not os.path.exists(tmp):
            print("%-6s 运行失败：%s" % (t, (r.stderr or r.stdout)[-200:]))
            continue
        logits = np.load(tmp)["logits"]
        os.remove(tmp)
        kl, top1 = kl_top1(ref, logits)
        results[t] = {"kl_mean": round(kl, 4), "top1_agreement": round(top1, 4)}
        print("%-6s KL %7.3f ｜ top-1 一致 %5.1f%%" % (t, kl, top1 * 100), flush=True)
        with open(args.out, "w") as fh:
            json.dump({"reference": os.path.basename(args.ref),
                       "suffix": args.suffix, "tokens": args.tokens,
                       "targets": results}, fh, ensure_ascii=False, indent=2)

    # 汇总
    ranked = sorted(results.items(), key=lambda kv: kv[1]["kl_mean"])
    print("\n==== 敏感度排序（KL 越小越安全）====")
    for name, v in ranked:
        print("%-6s KL %7.3f ｜ top-1 %5.1f%%" % (name, v["kl_mean"], v["top1_agreement"] * 100))
    print("\n已写出 %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
