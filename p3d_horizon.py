#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P3-D：把测量窗口 H 从 64 扩到 128，并在独立测试文本上复核分层排序。

协议 §5 要求"对可编译候选在 H=1、8、32、128 下估计指标"，而本轮 P3 只在 pilot 文本上
做过 H 分析、且最大窗口只有 64 步（协议要求的"跨分布复核"与 H=128 都缺失）。
本脚本一次补齐两件事：

  1) collect：用 128 步前缀收集 33 个目标（32 层 + 输出头）"只量化该目标"的臂，
     参考臂是同长度 fp16 臂；每目标独立子进程（避免同进程反复建 Engine 触发 245000）；
  2) analyze：在 pilot 与 p1test（P1 语料的独立测试分割，不参与校准）两份文本上
     分别做 H=1/8/32/64/128 的排序相关、top-k 重合率与配对 bootstrap；
  3) plan：用 H=8 排序选前 8 层做成计划，在 128 步窗口下实测 KL，并与线上
     P1 校准计划（plan_p1_top8.json）同窗口对照。

用法：
  python p3d_horizon.py --mode collect     # 只采数据（可中断续跑）
  python p3d_horizon.py --mode analyze     # 只做离线分析
  python p3d_horizon.py --mode plan        # 只做计划级实测
  python p3d_horizon.py                    # 三步顺序执行
输出：p3d_result.json；臂数据在 p3d_parts/。
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

MODEL_DIR = os.environ.get("RWKV_MODEL_DIR", "/home/disk/models/rwkv7-2.9b")
PY = os.environ.get("PY_NPU", "/home/disk/miniconda3/envs/npu22/bin/python")
SUFFIX = "_p1_q"
PARTS = os.path.join(MODEL_DIR, "p3d_parts")
TARGETS = ["head"] + [str(i) for i in range(32)]
TEXTS = ["pilot", "p1test"]
TOKENS = 128
HORIZONS = [1, 8, 32, 64, 128]
RESULT = os.path.join(MODEL_DIR, "p3d_result.json")


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


def arm_path(text, name):
    return os.path.join(PARTS, "arm_%s_%s.npz" % (text, name))


def ref_path(text):
    return os.path.join(MODEL_DIR, "arm_fp16_%d_%s.npz" % (TOKENS, text))


def run_arm(text, name, plan, out, decode_steps=8):
    cmd = [PY, os.path.join(MODEL_DIR, "ab_run_arm.py"), "--tokens", str(TOKENS),
           "--text", text, "--out", out, "--decode-steps", str(decode_steps),
           "--meta", out[:-4] + ".meta.json"]
    if plan is not None:
        cmd += ["--plan-json", json.dumps(plan)]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    ok = os.path.exists(out)
    # 把子进程的分阶段计时留档（每条臂一行），便于事后核对慢在哪里
    with open(os.path.join(MODEL_DIR, "p3d_arms.log"), "a") as fh:
        fh.write("=== %s %s %s\n" % (time.strftime("%F %T"), text, name))
        fh.write((r.stdout or "")[-2000:])
        if r.returncode != 0:
            fh.write("[stderr]\n" + (r.stderr or "")[-2000:])
    print("%-8s %-6s %s（%.0fs）%s"
          % (text, name, "OK" if ok else "失败", time.time() - t0,
             "" if ok else (r.stderr or r.stdout)[-200:]), flush=True)
    return ok


def collect(only_text=None):
    os.makedirs(PARTS, exist_ok=True)
    for text in ([only_text] if only_text else TEXTS):
        rp = ref_path(text)
        if os.path.exists(rp):
            print("跳过参考臂 %s（已有）" % os.path.basename(rp), flush=True)
        else:
            run_arm(text, "ref_fp16", None, rp)
        for name in TARGETS:
            out = arm_path(text, name)
            if os.path.exists(out):
                continue
            run_arm(text, name, target_plan(name), out)


def load_kls(text):
    ref = np.load(ref_path(text))["logits"]
    kls, n_short = {}, {}
    for name in TARGETS:
        path = arm_path(text, name)
        if not os.path.exists(path):
            continue
        logits = np.load(path)["logits"]
        n_short[name] = int(logits.shape[0])
        kls[name] = kl_per_step(ref, logits)
    return ref, kls, n_short


def analyze_text(text):
    ref, kls, n_short = load_kls(text)
    print("\n[%s] 目标 %d 个 ｜ 参考轨迹 %d 步 ｜ 各臂步数 %s"
          % (text, len(kls), ref.shape[0],
             sorted(set(n_short.values()))), flush=True)
    horizons = [h for h in HORIZONS if h <= ref.shape[0]]
    mean_kl = {h: {n: float(v[:h].mean()) for n, v in kls.items()} for h in horizons}

    full_h = horizons[-1]
    full = mean_kl[full_h]
    order_full = sorted(full, key=lambda n: full[n])
    names = list(full)
    corr, overlap = {}, {}
    for h in horizons[:-1]:
        corr[h] = round(spearman([mean_kl[h][n] for n in names],
                                 [full[n] for n in names]), 4)
        order_h = sorted(names, key=lambda n: mean_kl[h][n])
        overlap[h] = {("top%d" % k): round(len(set(order_h[:k]) & set(order_full[:k])) / k, 3)
                      for k in (8, 16, 33)}

    print("%-6s %-10s %s" % ("H", "Spearman", "top8/top16/top33 重合（对照 H=%d）" % full_h))
    for h in horizons[:-1]:
        o = overlap[h]
        print("%-6d %-10.3f %.2f / %.2f / %.2f"
              % (h, corr[h], o["top8"], o["top16"], o["top33"]), flush=True)

    # 配对 bootstrap：重采样 token 位置（H 步窗口内），看短窗口排序与全窗口排序的相关性
    rng = np.random.default_rng(0)
    n_tok = min(len(v) for v in kls.values())
    boot = {}
    for h in [x for x in (8, 32, 64) if x <= n_tok and x != full_h]:
        vals = []
        for _ in range(200):
            full_idx = rng.integers(0, n_tok, n_tok)
            short_idx = rng.integers(0, h, h)
            full_b = {nm: float(kls[nm][full_idx].mean()) for nm in names}
            short_b = {nm: float(kls[nm][:h][short_idx].mean()) for nm in names}
            vals.append(spearman([short_b[nm] for nm in names],
                                 [full_b[nm] for nm in names]))
        vals = np.array([v for v in vals if not np.isnan(v)])
        boot["H%d" % h] = {"mean": round(float(vals.mean()), 4),
                           "ci95": [round(float(np.percentile(vals, 2.5)), 4),
                                    round(float(np.percentile(vals, 97.5)), 4)],
                           "n_boot": int(len(vals))}
        print("bootstrap H=%-3d vs H=%d：Spearman %.3f（95%% CI %.3f~%.3f，%d 次）"
              % (h, full_h, boot["H%d" % h]["mean"], boot["H%d" % h]["ci95"][0],
                 boot["H%d" % h]["ci95"][1], boot["H%d" % h]["n_boot"]), flush=True)

    return {"targets": len(kls), "ref_steps": int(ref.shape[0]),
            "horizons": horizons,
            "kl_mean_by_horizon": {h: {n: round(x, 6) for n, x in mean_kl[h].items()}
                                   for h in horizons},
            "spearman_vs_full": corr, "topk_overlap_vs_full": overlap,
            "bootstrap_spearman": boot, "order_full": order_full,
            "order_h8": sorted(names, key=lambda n: mean_kl[8][n])}


def analyze():
    report = {"tokens": TOKENS, "texts": {}}
    for text in TEXTS:
        if not os.path.exists(ref_path(text)):
            print("缺参考臂 %s，跳过 %s" % (os.path.basename(ref_path(text)), text))
            continue
        report["texts"][text] = analyze_text(text)

    # 跨分布：pilot 与 p1test 的全窗口层排序是否一致
    if all(t in report["texts"] for t in TEXTS):
        def kl_at(text, h):
            table = report["texts"][text]["kl_mean_by_horizon"]
            return table[str(h)] if str(h) in table else table[h]

        full_h = max(report["texts"]["pilot"]["horizons"])
        a, b = kl_at("pilot", full_h), kl_at("p1test", full_h)
        names = [n for n in a if n in b]
        rho = round(spearman([a[n] for n in names], [b[n] for n in names]), 4)
        oa = report["texts"]["pilot"]["order_full"]
        ob = report["texts"]["p1test"]["order_full"]
        cross = {"spearman_full_rank": rho,
                 "top8_overlap": round(len(set(oa[:8]) & set(ob[:8])) / 8, 3),
                 "top16_overlap": round(len(set(oa[:16]) & set(ob[:16])) / 16, 3),
                 "pilot_top8": oa[:8], "p1test_top8": ob[:8]}
        report["cross_text"] = cross
        print("\n跨分布（全窗口 %d 步）：Spearman %.3f ｜ top8 重合 %.2f ｜ top16 重合 %.2f"
              % (full_h, rho, cross["top8_overlap"], cross["top16_overlap"]), flush=True)

    with open(RESULT, "w") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print("\n已写出 %s" % os.path.basename(RESULT), flush=True)
    return report


def plan_check():
    """H=8 选出的前 8 层（在 128 步窗口上重选）与线上 P1 计划，同窗口实测 KL。"""
    if os.path.exists(RESULT):
        with open(RESULT) as fh:
            rep = json.load(fh)
    else:
        rep = analyze()
    out = {}
    for text in TEXTS:
        if text not in rep["texts"]:
            continue
        order8 = rep["texts"][text]["order_h8"][:8]
        picked = [int(n) for n in order8 if n != "head"]
        plans = {
            "h8_from_%s" % text: {"suffix": SUFFIX, "layers": picked,
                                  "head": "head" in order8},
        }
        p1 = os.path.join(MODEL_DIR, "plan_p1_top8.json")
        if os.path.exists(p1):
            with open(p1) as fh:
                plans["p1_plan_top8"] = json.load(fh)
        for tag, plan in plans.items():
            arm = os.path.join(MODEL_DIR, "arm_%s_%s_%d.npz" % (tag, text, TOKENS))
            if not os.path.exists(arm):
                run_arm(text, tag, plan, arm, decode_steps=8)
            ref = np.load(ref_path(text))["logits"]
            logits = np.load(arm)["logits"]
            kl = kl_per_step(ref, logits)
            key = "%s@%s" % (tag, text)
            out[key] = {"plan": plan, "kl_mean": round(float(kl.mean()), 6),
                        "kl_mean_h8": round(float(kl[:8].mean()), 6),
                        "kl_mean_h64": round(float(kl[:64].mean()), 6),
                        "steps": int(logits.shape[0])}
            print("%-28s 全窗口 KL %.4f ｜ H=8 %.4f ｜ H=64 %.4f"
                  % (key, out[key]["kl_mean"], out[key]["kl_mean_h8"],
                     out[key]["kl_mean_h64"]), flush=True)
    rep = rep if isinstance(rep, dict) else {}
    rep["plans"] = out
    with open(RESULT, "w") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=2)
    return rep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="all",
                        choices=["all", "collect", "analyze", "plan"])
    parser.add_argument("--text", default=None, help="只采某一份文本的臂")
    args = parser.parse_args()
    if args.mode in ("all", "collect"):
        collect(args.text)
    if args.mode in ("all", "analyze"):
        analyze()
    if args.mode in ("all", "plan"):
        plan_check()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
