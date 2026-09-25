#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P2-D：单次量化脉冲（无状态传播）能否预测持续量化（有传播）的结果？

协议 §4 的 D 组：同一前缀、同一候选，比较
  * 脉冲（pulse）：每步执行前把 fp16 参考状态灌回 → 只剩**单步输出偏差**，不累积；
  * 持续（sustained）：量化模型保留自己的状态 → 单步偏差 + **状态传播**。
在 8 个候选上各测两种指标，算 Spearman 排序相关与倍数关系。

**每个候选跑在独立子进程里**：ACL context 与 .om 的设备内存要到进程退出才彻底释放，
同一进程里反复 Engine() 会在第二个候选就报 245000（本脚本第一版就是这么挂的）。

用法:
  python p2d_pulse.py --ref-dir p2c_ref --out p2d_result.json     # 驱动全流程
  python p2d_pulse.py --ref-dir p2c_ref --candidate p1_top8 --out-part parts/p1_top8.json
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

MODEL_DIR = "/home/disk/models/rwkv7-2.9b"
sys.path.insert(0, MODEL_DIR)
PY = "/home/disk/miniconda3/envs/npu22/bin/python"

CANDIDATES = [
    {"name": "p1_top8", "plan": "plan_p1_top8.json"},
    {"name": "p1_top16", "plan": "plan_p1_top16.json"},
    {"name": "p1_top24", "plan": "plan_p1_top24.json"},
    {"name": "p1_full", "plan": "plan_p1_top33.json"},
    {"name": "cp_top8", "plan": "plan_top8.json"},
    {"name": "cp_full", "suffix": "_cp_q"},
    {"name": "p1_layer9_only", "suffix": "_p1_q", "layers": [9], "head": False},
    {"name": "p1_head_only", "suffix": "_p1_q", "layers": [], "head": True},
]


def softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def kl_list(ref_logits, logits):
    out = []
    for t in range(logits.shape[0]):
        p, q = softmax(ref_logits[t]), softmax(logits[t])
        out.append(float(np.sum(p * (np.log(p + 1e-12) - np.log(q + 1e-12)))))
    return out


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


def load_ref(ref_dir):
    ids = np.load(os.path.join(ref_dir, "ids.npy")).tolist()
    logits = np.load(os.path.join(ref_dir, "logits.npy"))
    keys = [os.path.basename(p)[:-4] for p in sorted(glob.glob(os.path.join(ref_dir, "*.npy")))
            if os.path.basename(p) not in ("logits.npy", "ids.npy", "shape.npy")]
    states = {k: np.load(os.path.join(ref_dir, k + ".npy"), mmap_mode="r") for k in keys}
    return ids, logits, states


def run(engine, ids, states=None):
    engine.reset_state()
    out = []
    for t, tok in enumerate(ids):
        if t > 0 and states is not None:
            engine.import_state({k: np.asarray(v[t - 1]) for k, v in states.items()})
        out.append(engine.step(int(tok)).copy())
    return np.stack(out)


def run_candidate(cand, ref_dir):
    from rwkv7_serve2 import Engine

    ids, ref_logits, states = load_ref(ref_dir)
    if cand.get("plan"):
        plan = json.load(open(os.path.join(MODEL_DIR, cand["plan"])))
        engine = Engine(MODEL_DIR, plan=plan)
    elif "layers" in cand:
        plan = {"suffix": cand.get("suffix", "_p1_q"),
                "layers": cand["layers"], "head": cand.get("head", False)}
        engine = Engine(MODEL_DIR, suffix=cand.get("suffix", "_p1_q"), plan=plan)
    else:
        engine = Engine(MODEL_DIR, suffix=cand.get("suffix", "_cp_q"))

    t0 = time.time()
    pulse = run(engine, ids, states)
    sustained = run(engine, ids, None)
    kl_p, kl_s = kl_list(ref_logits, pulse), kl_list(ref_logits, sustained)
    return {
        "n_int8": engine.n_quant,
        "pulse_kl_mean": round(float(np.mean(kl_p)), 4),
        "sustained_kl_mean": round(float(np.mean(kl_s)), 4),
        "ratio": round(float(np.mean(kl_s) / max(np.mean(kl_p), 1e-9)), 3),
        "pulse_kl_last8": round(float(np.mean(kl_p[-8:])), 4),
        "sustained_kl_last8": round(float(np.mean(kl_s[-8:])), 4),
        "seconds": round(time.time() - t0, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref-dir", default=os.path.join(MODEL_DIR, "p2c_ref"))
    parser.add_argument("--out", default=os.path.join(MODEL_DIR, "p2d_result.json"))
    parser.add_argument("--candidate", default=None, help="只跑这一个候选（子进程模式）")
    parser.add_argument("--out-part", default=None)
    args = parser.parse_args()

    if args.candidate:
        cand = next(c for c in CANDIDATES if c["name"] == args.candidate)
        entry = run_candidate(cand, args.ref_dir)
        with open(args.out_part, "w") as fh:
            json.dump({"name": cand["name"], **entry}, fh, ensure_ascii=False, indent=2)
        print("%-16s int8 %2d｜脉冲 %7.4f｜持续 %7.4f｜倍数 %5.2f｜%.0fs"
              % (cand["name"], entry["n_int8"], entry["pulse_kl_mean"],
                 entry["sustained_kl_mean"], entry["ratio"], entry["seconds"]), flush=True)
        return 0

    # 驱动模式：每个候选一个子进程，避免设备内存累积
    parts_dir = os.path.join(MODEL_DIR, "p2d_parts")
    os.makedirs(parts_dir, exist_ok=True)
    report = {"candidates": {}}
    for cand in CANDIDATES:
        part = os.path.join(parts_dir, cand["name"] + ".json")
        if not os.path.exists(part):
            cmd = [PY, os.path.join(MODEL_DIR, "p2d_pulse.py"),
                   "--ref-dir", args.ref_dir, "--candidate", cand["name"], "--out-part", part]
            r = subprocess.run(cmd, capture_output=True, text=True)
            print((r.stdout or "").strip() or ("%s 失败：%s" % (cand["name"], (r.stderr or "")[-300:])),
                  flush=True)
        if os.path.exists(part):
            with open(part) as fh:
                d = json.load(fh)
            d.pop("name", None)
            report["candidates"][cand["name"]] = d

    names = list(report["candidates"])
    if len(names) >= 3:
        xs = [report["candidates"][n]["pulse_kl_mean"] for n in names]
        ys = [report["candidates"][n]["sustained_kl_mean"] for n in names]
        report["spearman_pulse_vs_sustained"] = round(spearman(xs, ys), 4)
        report["ratio_mean"] = round(
            float(np.mean([report["candidates"][n]["ratio"] for n in names])), 3)
        print("\n脉冲 vs 持续 Spearman 排序相关 %.3f｜倍数均值 %.2f×（%d 个候选）"
              % (report["spearman_pulse_vs_sustained"], report["ratio_mean"], len(names)),
              flush=True)
    report["tokens"] = int(np.load(os.path.join(args.ref_dir, "logits.npy")).shape[0])
    with open(args.out, "w") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print("已写出 %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
