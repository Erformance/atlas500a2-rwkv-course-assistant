#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按敏感度构造混合精度候选计划，测出"质量—速度"帕累托曲线。

输入：sensitivity.json（逐层敏感度，sensitivity.py 产出）+ arm_fp16.npz（fp16 参考）
做法：把"层"和"输出头"统一视为可量化单元，按 KL 从小到大排序；
      取前 N 个最安全的单元做 int8、其余保 fp16，N 取若干档位，逐个实测：
      与 fp16 的 KL、top-1 一致率、decode 速度。
输出：pareto.json + 控制台表格

用法: python pareto.py --sizes 8,16,24,28,33
"""

import argparse
import json
import os
import subprocess

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


def run_plan(plan, tokens, out_npz):
    cmd = [PY, os.path.join(MODEL_DIR, "ab_run_arm.py"),
           "--plan-json", json.dumps(plan), "--tokens", str(tokens), "--out", out_npz]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if not os.path.exists(out_npz):
        print("  运行失败：%s" % ((r.stderr or r.stdout)[-200:],))
        return None, None
    data = np.load(out_npz)
    logits = data["logits"]
    os.remove(out_npz)
    # 复跑一次取速度（ab_run_arm 的 meta 只写文件；这里从 stdout 解析更省事）
    speed = None
    for line in (r.stdout or "").splitlines():
        if "ms/token" in line:
            try:
                speed = float(line.split("（")[-1].split(" tok/s")[0])
            except Exception:
                pass
    return logits, speed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", default="_cp_q")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--sizes", default="8,16,24,28,33")
    parser.add_argument("--ref", default=os.path.join(MODEL_DIR, "arm_fp16.npz"))
    parser.add_argument("--sens", default=os.path.join(MODEL_DIR, "sensitivity.json"))
    parser.add_argument("--out", default=os.path.join(MODEL_DIR, "pareto.json"))
    args = parser.parse_args()

    ref = np.load(args.ref)["logits"]
    sens = json.load(open(args.sens))["targets"]
    ranked = sorted(sens.items(), key=lambda kv: kv[1]["kl_mean"])
    print("敏感度排序（最安全在前）：")
    for name, v in ranked[:6]:
        print("  %-6s KL %.4f" % (name, v["kl_mean"]))
    print("  …")
    for name, v in ranked[-4:]:
        print("  %-6s KL %.4f" % (name, v["kl_mean"]))

    report = {"ranked": [n for n, _ in ranked], "plans": {}}
    print("\n%-8s %-8s %-10s %-10s %s" % ("计划", "int8 单元", "KL 均值", "top-1", "decode"))
    for size in [int(x) for x in args.sizes.split(",")]:
        picked = [n for n, _ in ranked[:size]]
        layers = [int(n) for n in picked if n != "head"]
        plan = {"suffix": args.suffix, "layers": layers, "head": "head" in picked}
        tmp = "/tmp/arm_pareto_%d.npz" % size
        logits, speed = run_plan(plan, args.tokens, tmp)
        if logits is None:
            continue
        kl, top1 = kl_top1(ref, logits)
        entry = {"int8_units": size, "layers": layers, "head": plan["head"],
                 "kl_mean": round(kl, 4), "top1_agreement": round(top1, 4),
                 "decode_tok_s": speed}
        report["plans"]["top%d" % size] = entry
        print("%-8s %-8d %-10.3f %-10.1f%% %s tok/s"
              % ("top%d" % size, size, kl, top1 * 100,
                 ("%.2f" % speed) if speed else "?"))
        with open(args.out, "w") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)

    print("\n已写出 %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
