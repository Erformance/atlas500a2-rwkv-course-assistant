#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按两个分布的敏感度取交集，构造"不偏科"的混合精度计划并实测。

动机：只用自然文本选的层（top16）在代码题上会翻车，因为敏感度依赖数据分布。
做法：对每个可量化单元取 max(KL_自然文本, KL_代码文本) 作为"最坏情况敏感度"，
      按它从小到大排序取前 N 个 → 得到一个在两个分布上都安全的计划。
对照：把"自然文本选出的 top16"也拿到代码文本上测一遍，量化其偏科程度。

用法: python plan_both.py --size 16
输出: plan_top{N}_both.json、plan_both_result.json
"""

import argparse
import json
import os
import subprocess

import numpy as np

MODEL_DIR = "/home/disk/models/rwkv7-2.9b"
PY = "/home/disk/miniconda3/envs/np22/bin/python"
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


def run_plan(plan, text, ref_npz, tag):
    tmp = "/tmp/arm_%s.npz" % tag
    cmd = [PY, os.path.join(MODEL_DIR, "ab_run_arm.py"),
           "--plan-json", json.dumps(plan), "--tokens", "64", "--text", text,
           "--out", tmp]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if not os.path.exists(tmp):
        print("  %s 运行失败：%s" % (tag, (r.stderr or r.stdout)[-200:]))
        return None, None, None
    logits = np.load(tmp)["logits"]
    os.remove(tmp)
    ref = np.load(ref_npz)["logits"]
    kl, top1 = kl_top1(ref, logits)
    speed = None
    for line in (r.stdout or "").splitlines():
        if "tok/s" in line:
            try:
                speed = float(line.split("（")[-1].split(" tok/s")[0])
            except Exception:
                pass
    return kl, top1, speed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=16)
    parser.add_argument("--suffix", default="_cp_q")
    args = parser.parse_args()

    pilot = json.load(open(os.path.join(MODEL_DIR, "sensitivity.json")))["targets"]
    mixed = json.load(open(os.path.join(MODEL_DIR, "sensitivity_mixed.json")))["targets"]
    common = [k for k in pilot if k in mixed]
    worst = {k: max(pilot[k]["kl_mean"], mixed[k]["kl_mean"]) for k in common}
    ranked = sorted(worst.items(), key=lambda kv: kv[1])
    picked = [n for n, _ in ranked[: args.size]]
    layers = [int(n) for n in picked if n != "head"]
    plan = {"suffix": args.suffix, "layers": layers, "head": "head" in picked}

    print("按两分布最坏情况排序（最安全在前）：")
    for n, v in ranked[:6]:
        print("  %-6s 自然 %.3f / 代码 %.3f" % (n, pilot[n]["kl_mean"], mixed[n]["kl_mean"]))
    print("  …")
    for n, v in ranked[-4:]:
        print("  %-6s 自然 %.3f / 代码 %.3f" % (n, pilot[n]["kl_mean"], mixed[n]["kl_mean"]))
    with open(os.path.join(MODEL_DIR, "plan_top%d_both.json" % args.size), "w") as fh:
        json.dump(plan, fh, indent=2)

    ref_pilot = os.path.join(MODEL_DIR, "arm_fp16.npz")
    ref_mixed = os.path.join(MODEL_DIR, "arm_fp16_mixed.npz")
    old_plan_path = os.path.join(MODEL_DIR, "plan_top16.json")

    report = {"size": args.size, "layers": layers, "head": plan["head"], "runs": {}}
    print("\n%-22s %-12s %-12s %s" % ("计划", "自然文本 KL", "代码文本 KL", "decode"))
    for tag, p, f in (("交集 top%d" % args.size, plan, None),
                      ("自然文本 top16（对照）",
                       json.load(open(old_plan_path)) if os.path.exists(old_plan_path) else None,
                       None)):
        if p is None:
            continue
        kp, tp, sp = run_plan(p, "pilot", ref_pilot, "pilot_%s" % tag.replace(" ", "_"))
        km, tm, sm = run_plan(p, "mixed", ref_mixed, "mixed_%s" % tag.replace(" ", "_"))
        report["runs"][tag] = {"kl_pilot": kp, "top1_pilot": tp,
                               "kl_mixed": km, "top1_mixed": tm, "decode_tok_s": sp or sm}
        print("%-22s %-12s %-12s %s"
              % (tag,
                 "-" if kp is None else "%.4f" % kp,
                 "-" if km is None else "%.4f" % km,
                 "-" if not (sp or sm) else "%.2f tok/s" % (sp or sm)))

    with open(os.path.join(MODEL_DIR, "plan_both_result.json"), "w") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print("\n已写出 plan_both_result.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
