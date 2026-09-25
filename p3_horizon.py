#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P3（窄版）：测量窗口 H 取多长才够？——协议 §5 的核心问题落到可测量版本。

协议 §5 问"多步/方向指标是否具有可重复的增量价值、能否在匹配搜索预算下改善计划"。
在 2.9B 上我们把它做成：**把逐层敏感度的测量窗口从 64 步缩到 32/8/1 步，
还能不能把层排序对、并选出同样好的混合精度计划？**

步骤：
  1) collect：33 个目标（32 层 + 输出头）各跑一条"只量化该目标"的臂，保存逐步 logits
     （每个目标独立子进程，避免设备内存累积；parts 目录可续跑）；
  2) analyze（离线，不需要 NPU）：用同一条 fp16 参考轨迹算每个目标逐步 KL，
     再截断到 H=1/8/32/64，比较各 H 的 Spearman 排序相关与 top-k 重合率（k=8/16）；
  3) plan：用 H=1 / H=8 / H=64 三个排序各取前 8 层，**实测**三个计划的 KL
     （层已全部编译为 `_p1_q`，零编译成本）。

用法：
  python p3_horizon.py                 # 全部三步
  python p3_horizon.py --mode collect  # 只采数据
  python p3_horizon.py --mode analyze  # 只做离线分析
  python p3_horizon.py --mode plan     # 只做计划级验证
输出：p3_result.json
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
PY = "/home/disk/miniconda3/envs/npu22/bin/python"
SUFFIX = "_p1_q"
PARTS = os.path.join(MODEL_DIR, "p3_parts")
TARGETS = ["head"] + [str(i) for i in range(32)]
HORIZONS = [1, 8, 32, 64]


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


def target_plan(name):
    if name == "head":
        return {"suffix": SUFFIX, "layers": [], "head": True}
    return {"suffix": SUFFIX, "layers": [int(name)], "head": False}


def collect():
    os.makedirs(PARTS, exist_ok=True)
    for name in TARGETS:
        out = os.path.join(PARTS, "arm_%s.npz" % name)
        if os.path.exists(out):
            print("跳过 %s（已有）" % name, flush=True)
            continue
        cmd = [PY, os.path.join(MODEL_DIR, "ab_run_arm.py"),
               "--plan-json", json.dumps(target_plan(name)),
               "--tokens", "64", "--text", "pilot", "--out", out]
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True)
        ok = os.path.exists(out)
        print("%-6s %s（%.0fs）%s" % (name, "OK" if ok else "失败",
                                     time.time() - t0,
                                     "" if ok else (r.stderr or r.stdout)[-160:]),
              flush=True)


def load_kls():
    ref = np.load(os.path.join(MODEL_DIR, "arm_fp16.npz"))["logits"]
    kls = {}
    for name in TARGETS:
        path = os.path.join(PARTS, "arm_%s.npz" % name)
        if not os.path.exists(path):
            continue
        logits = np.load(path)["logits"]
        kls[name] = kl_per_step(ref, logits)
    return ref, kls


def analyze():
    ref, kls = load_kls()
    print("可用目标 %d 个，参考轨迹 %d 步" % (len(kls), ref.shape[0]), flush=True)
    mean_kl = {h: {n: float(v[:h].mean()) for n, v in kls.items()} for h in HORIZONS}

    # 保留上一次跑出来的计划级结果（analyze 会重写这个文件）
    old_path = os.path.join(MODEL_DIR, "p3_result.json")
    old = {}
    if os.path.exists(old_path):
        try:
            with open(old_path) as fh:
                old = json.load(fh)
        except Exception:
            old = {}
    report = {"targets": len(kls), "horizons": HORIZONS,
              "kl_mean_by_horizon": {h: {n: round(x, 6) for n, x in mean_kl[h].items()}
                                     for h in HORIZONS}}
    if "plans_from_each_horizon" in old:
        report["plans_from_each_horizon"] = old["plans_from_each_horizon"]

    # 各 H 与 H=64 的排序相关性 + top-k 重合
    full = mean_kl[64]
    order_full = sorted(full, key=lambda n: full[n])
    corr, overlap = {}, {}
    for h in HORIZONS[:-1]:
        names = list(full)
        corr[h] = round(spearman([mean_kl[h][n] for n in names],
                                 [full[n] for n in names]), 4)
        order_h = sorted(names, key=lambda n: mean_kl[h][n])
        overlap[h] = {("top%d" % k): len(set(order_h[:k]) & set(order_full[:k])) / k
                      for k in (8, 16, 33)}
    report["spearman_vs_h64"] = corr
    report["topk_overlap_vs_h64"] = overlap

    print("\n%-6s %-10s %s" % ("H", "Spearman", "top8/top16/top33 重合率"))
    for h in HORIZONS[:-1]:
        o = overlap[h]
        print("%-6d %-10.3f %.2f / %.2f / %.2f"
              % (h, corr[h], o["top8"], o["top16"], o["top33"]), flush=True)

    print("\n各 H 下最安全的 8 个目标：")
    for h in HORIZONS:
        order = sorted(mean_kl[h], key=lambda n: mean_kl[h][n])
        print("  H=%-3d %s" % (h, ",".join(order[:8])), flush=True)

    # 配对 bootstrap：重采样 token 位置，看"短窗口排序 vs 64 步排序"的相关性稳不稳。
    # H=1 只有一步、无法重采样，只对 H=8/32 做。
    rng = np.random.default_rng(0)
    n_tok = min(len(v) for v in kls.values())
    names = list(kls)
    boot = {}
    for h in (8, 32):
        vals = []
        for _ in range(200):
            idx = rng.integers(0, n_tok, n_tok)
            full = {nm: float(kls[nm][idx].mean()) for nm in names}
            short_idx = rng.integers(0, h, h)
            short = {nm: float(kls[nm][:h][short_idx].mean()) for nm in names}
            vals.append(spearman([short[nm] for nm in names], [full[nm] for nm in names]))
        vals = np.array([v for v in vals if not np.isnan(v)])
        boot["H%d" % h] = {"mean": round(float(vals.mean()), 4),
                           "ci95": [round(float(np.percentile(vals, 2.5)), 4),
                                    round(float(np.percentile(vals, 97.5)), 4)],
                           "n_boot": int(len(vals))}
        print("配对 bootstrap：H=%-3d vs H=64 的 Spearman %.3f（95%% CI %.3f~%.3f，%d 次）"
              % (h, boot["H%d" % h]["mean"], boot["H%d" % h]["ci95"][0],
                 boot["H%d" % h]["ci95"][1], boot["H%d" % h]["n_boot"]), flush=True)
    report["bootstrap_spearman"] = boot

    with open(os.path.join(MODEL_DIR, "p3_result.json"), "w") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print("\n已写出 p3_result.json")
    return report


def plan_check():
    """用 H=1 / H=8 / H=64 三个排序各取前 8 层，实测计划 KL（层都已编译，零成本）。"""
    _, kls = load_kls()
    mean_kl = {h: {n: float(v[:h].mean()) for n, v in kls.items()} for h in HORIZONS}
    ref_logits = np.load(os.path.join(MODEL_DIR, "arm_fp16.npz"))["logits"]
    out = {}
    for h in (1, 8, 64):
        order = sorted(mean_kl[h], key=lambda n: mean_kl[h][n])
        picked = [n for n in order[:8] if n != "head"]
        plan = {"suffix": SUFFIX, "layers": [int(x) for x in picked],
                "head": "head" in order[:8]}
        plan_path = os.path.join(MODEL_DIR, "p3_plan_h%d.json" % h)
        with open(plan_path, "w") as fh:
            json.dump(plan, fh, ensure_ascii=False, indent=2)
        arm = os.path.join(PARTS, "plan_h%d.npz" % h)
        cmd = [PY, os.path.join(MODEL_DIR, "ab_run_arm.py"),
               "--plan-json", json.dumps(plan), "--tokens", "64", "--text", "pilot",
               "--out", arm, "--meta", os.path.join(PARTS, "meta_h%d.json" % h)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if not os.path.exists(arm):
            print("H=%d 计划实测失败：%s" % (h, (r.stderr or r.stdout)[-200:]), flush=True)
            continue
        logits = np.load(arm)["logits"]
        kl = kl_per_step(ref_logits, logits)
        out["H%d" % h] = {"layers": plan["layers"], "head": plan["head"],
                          "kl_mean": round(float(kl.mean()), 4),
                          "kl_first8": round(float(kl[:8].mean()), 4),
                          "top1_agreement": round(float(np.mean(
                              logits.argmax(axis=1) == ref_logits.argmax(axis=1))), 4)}
        print("%-5s 前 8 层 %s｜计划 KL %7.4f｜top-1 %.1f%%"
              % ("H=%d" % h, plan["layers"], out["H%d" % h]["kl_mean"],
                 out["H%d" % h]["top1_agreement"] * 100), flush=True)

    path = os.path.join(MODEL_DIR, "p3_result.json")
    report = json.load(open(path)) if os.path.exists(path) else {}
    report["plans_from_each_horizon"] = out
    with open(path, "w") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print("计划级结果已并入 p3_result.json")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="all",
                        choices=["all", "collect", "analyze", "plan"])
    args = parser.parse_args()
    if args.mode in ("all", "collect"):
        collect()
    if args.mode in ("all", "analyze"):
        analyze()
    if args.mode in ("all", "plan"):
        plan_check()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
