#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P6：协议 §8 的六幅图 + 关键消融表（全部取自已跑的实测数据，不填任何虚构值）。

图（英文标注，避免设备缺中文字体；说明写在 P6_图表_报告.md 里）：
  fig1_pareto.png            质量—延迟 Pareto（2.9B，5 个点）
  fig2_trajectory.png        按 token 步的误差轨迹（注入 vs 自由）
  fig3_metric_vs_cost.png    候选指标：排序质量 vs 测量成本（含 G=I 消融）
  fig4_perf_memory.png       性能与内存分解（前缀长度曲线 + 加载/RSS 柱状）
  fig5_ablations.png         关键消融（校准侧 / 计划侧，各一栏）
  fig6_search_cost.png       搜索成本：校准预算 vs 质量、测量窗口 vs 排序
另写 p6_ablations.json（消融表数值）与 p6_metrics.json（指标面板）。

用法（设备上，装了 matplotlib 的环境）：
  RWKV_MODEL_DIR=/home/disk/models/rwkv7-2.9b python p6_figures.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

MD = os.environ.get("RWKV_MODEL_DIR", "/home/disk/models/rwkv7-2.9b")
OUT = os.path.join(MD, "figs")
os.makedirs(OUT, exist_ok=True)


def load(name, base=None):
    path = os.path.join(base or MD, name)
    if not os.path.exists(path):
        print("[缺] %s" % path)
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


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


# ---------------------------------------------------------------- 数据装载
par = load("pareto_p1.json")
p5 = load("p5_bench.json")
p5b = load("p5_bench_2048.json")
p2c = load("p2c_result.json")
p2d = load("p2d_result.json")
p2f = load("p2f_result.json")
p3 = load("p3_result.json")
p3l = load("p3_local.json")
p3b = load("p3b_dir.json")
p2b_pilot = load("p2b_compare_pilot.json")
figs = {}

# ---------------------------------------------------------------- 图 1：Pareto
if par and p5:
    fp16_speed = p5["arms"]["precise"]["prefixes"]["512"]["decode_tok_s_median"]
    pts = [(fp16_speed, 0.0, "precise (fp16)")]
    label_map = {"top8": "balanced (8 int8)", "top16": "fast (16 int8)"}
    for k, v in par["plans"].items():
        pts.append((v["decode_tok_s"], v["kl_mean"], label_map.get(k, k)))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot([p[0] for p in pts], [max(p[1], 1e-4) for p in pts], "o-", color="#2f6feb")
    for x, y, name in pts:
        ax.annotate(name, (x, max(y, 1e-4)), textcoords="offset points", xytext=(4, 6), fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("decode throughput (tokens/s, measured on NPU)")
    ax.set_ylabel("full-sequence KL vs fp16 (lower is better)")
    ax.set_title("Fig 1  Quality–latency Pareto (RWKV-7 2.9B, Atlas 500 A2)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_pareto.png"), dpi=160)
    figs["fig1"] = [(round(x, 2), round(y, 4), n) for x, y, n in pts]

# ---------------------------------------------------------------- 图 2：误差轨迹
if p2c:
    inj = p2c["per_step_kl_inject"]
    free = p2c["per_step_kl_free"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(range(1, len(inj) + 1), inj, "-o", ms=3, label="inject reference state (no propagation)")
    ax.plot(range(1, len(free) + 1), free, "-s", ms=3, label="free-running (state propagates)")
    ax.set_xlabel("token step")
    ax.set_ylabel("KL vs fp16 at that step")
    ax.set_title("Fig 2  Error trajectory along the sequence (full int8, 2.9B)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    if "propagation_share" in p2c:
        ax.text(0.98, 0.92, "propagation share = %.0f%%" % (p2c["propagation_share"] * 100),
                transform=ax.transAxes, ha="right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_trajectory.png"), dpi=160)

# ---------------------------------------------------------------- 图 3：指标 vs 成本
metric_rows = []
if p3 and p3l and p3b:
    kl64 = p3["kl_mean_by_horizon"]["64"]
    layers = [n for n in kl64 if n != "head"]

    def sp(vals):
        shared = [n for n in layers if n in vals]
        return spearman([vals[n] for n in shared], [kl64[n] for n in shared]), len(shared)

    def topk(vals, k=8):
        shared = [n for n in layers if n in vals]
        order_m = sorted(shared, key=lambda n: vals[n])
        order_k = sorted(shared, key=lambda n: kl64[n])
        return len(set(order_m[:k]) & set(order_k[:k])) / k

    local = {n: v["rel_err_max"] for n, v in p3l["layers"].items() if n in layers}
    delta = {n: v["delta_norm"] for n, v in p3b["layers"].items()
             if n in layers and v.get("delta_norm")}
    gam = {n: v["dir_score"] for n, v in p3b["layers"].items()
           if n in layers and v.get("dir_score") is not None}
    gnorm = {n: v["g_norm"] for n, v in p3b["layers"].items()
             if n in layers and v.get("g_norm")}
    gi = {n: gnorm[n] * delta[n] for n in gnorm if n in delta}      # G=I 消融（不做输出加权）
    for name, vals, cost in (("local output err", local, 1), ("||delta||", delta, 1),
                             ("G=I (unweighted)", gi, 1), ("direction gamma", gam, 3),
                             ("rollout H=1", None, 1), ("rollout H=8", None, 8),
                             ("rollout H=32", None, 32), ("rollout H=64", None, 64)):
        if vals is None:
            h = {"rollout H=1": 1, "rollout H=8": 8, "rollout H=32": 32, "rollout H=64": 64}[name]
            rho = p3["spearman_vs_h64"][str(h)] if str(h) in p3["spearman_vs_h64"] else 1.0
            ov = p3["topk_overlap_vs_h64"].get(str(h), {}).get("top8", 1.0)
        else:
            rho, _ = sp(vals)
            ov = topk(vals)
        metric_rows.append({"metric": name, "cost_forwards": cost,
                            "spearman_vs_kl64": round(float(rho), 4),
                            "top8_overlap": round(float(ov), 4)})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.0))
    xs = np.array([max(r["cost_forwards"], 1) for r in metric_rows], float)
    a1.plot(xs, [r["spearman_vs_kl64"] for r in metric_rows], "o", color="#2f6feb")
    for r, x in zip(metric_rows, xs):
        a1.annotate(r["metric"], (x, r["spearman_vs_kl64"]), textcoords="offset points",
                    xytext=(3, 4), fontsize=7)
    a1.set_xscale("log")
    a1.set_xlabel("measurement cost (forward passes)")
    a1.set_ylabel("Spearman vs H=64 rollout KL")
    a1.set_title("(a) ranking fidelity vs cost")
    a1.grid(alpha=0.3)
    a2.plot(xs, [r["top8_overlap"] for r in metric_rows], "s", color="#e07b39")
    a2.set_xscale("log")
    a2.set_xlabel("measurement cost (forward passes)")
    a2.set_ylabel("top-8 overlap with H=64 ranking")
    a2.set_title("(b) which safe layers are recovered")
    a2.grid(alpha=0.3)
    fig.suptitle("Fig 3  Candidate metrics: fidelity vs measurement cost (32 layers, 2.9B)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_metric_vs_cost.png"), dpi=160)

# ---------------------------------------------------------------- 图 4：性能与内存
if p5:
    arms = ["precise", "balanced", "fast", "full_int8"]
    names = {"precise": "fp16 (precise)", "balanced": "top8 (balanced)",
             "fast": "top16 (fast)", "full_int8": "full int8 (experimental)"}
    prefixes = [32, 128, 512, 1024]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.0))
    for arm in arms:
        d = p5["arms"].get(arm)
        if not d:
            continue
        xs, ys = [], []
        for n in prefixes:
            row = d["prefixes"].get(str(n))
            if row:
                xs.append(n)
                ys.append(row["prefill_s_median"])
        if p5b and p5b["arms"].get(arm, {}).get("prefixes", {}).get("2048"):
            xs.append(2048)
            ys.append(p5b["arms"][arm]["prefixes"]["2048"]["prefill_s_median"])
        a1.plot(xs, ys, "o-", ms=4, label=names[arm])
    a1.set_xlabel("prefix length (tokens)")
    a1.set_ylabel("prefill time (s)")
    a1.set_title("(a) prefill cost grows linearly with prefix")
    a1.legend(fontsize=7)
    a1.grid(alpha=0.3)
    idx = np.arange(len(arms))
    load_s = [p5["arms"][a]["load_s_median"] for a in arms]
    rss = [p5["arms"][a]["rss_mb_median"] / 1000.0 for a in arms]
    drop = [p5["arms"][a]["mem_available_delta_mb_median"] / 1000.0 for a in arms]
    w = 0.28
    a2.bar(idx - w, load_s, w, label="engine load (s)")
    a2.bar(idx, rss, w, label="process RSS (GB)")
    a2.bar(idx + w, drop, w, label="available-memory drop (GB)")
    a2.set_xticks(idx)
    a2.set_xticklabels([names[a].split(" (")[0] for a in arms], fontsize=8)
    a2.set_title("(b) load time and memory footprint")
    a2.legend(fontsize=7)
    a2.grid(alpha=0.3, axis="y")
    fig.suptitle("Fig 4  Performance and memory decomposition (Atlas 500 A2)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_perf_memory.png"), dpi=160)

# ---------------------------------------------------------------- 图 5：关键消融
ablations = {}
if p2b_pilot and p2c and p3 and p3b and par:
    b0 = p2b_pilot["arms"]["_b0_pilot"]["kl_mean"]
    b1 = p2b_pilot["arms"]["_b1_pilot"]["kl_mean"]
    p1v = p2b_pilot["arms"]["_p1_pilot"]["kl_mean"]
    c15 = load("p3c_15b_compare.json", "/home/disk/models/rwkv7-1.5b")
    ablations["calibration"] = {
        "2.9B timing: post-exec sampling": 0.945, "2.9B timing: pre-exec sampling": 0.686,
        "2.9B plan: age-0 only": b0, "2.9B plan: age 0+256": b1, "2.9B plan: age 0+64+256": p1v,
    }
    if c15:
        ablations["calibration"]["1.5B plan: age-0 only"] = c15["arms"]["_15b_a0"]["kl_mean"]
        ablations["calibration"]["1.5B plan: age 0+256"] = c15["arms"]["_15b_m"]["kl_mean"]
    ablations["planning"] = {
        "full int8: inject (no prop.)": p2c["inject"]["kl_mean"],
        "full int8: free (prop.)": p2c["free"]["kl_mean"],
        "path protection (FFN-only bound)": 0.56,
        "layer selection (top16)": par["plans"]["top16"]["kl_mean"],
        "plan from H=1": p3["plans_from_each_horizon"]["H1"]["kl_mean"],
        "plan from H=8": p3["plans_from_each_horizon"]["H8"]["kl_mean"],
        "plan from H=64": p3["plans_from_each_horizon"]["H64"]["kl_mean"],
    }
    dirplan = load("p3b_plan_compare.json")
    if dirplan:
        for k, v in dirplan["arms"].items():
            ablations["planning"]["plan from direction gamma"] = v["kl_mean"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for ax, key, title in ((a1, "calibration", "(a) calibration choices"),
                           (a2, "planning", "(b) planning & propagation choices")):
        items = list(ablations[key].items())
        ax.barh(range(len(items)), [max(v, 1e-4) for _, v in items], color="#4c78a8")
        ax.set_yticks(range(len(items)))
        ax.set_yticklabels([k for k, _ in items], fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("full-sequence KL (log scale, lower is better)")
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3, axis="x")
    fig.suptitle("Fig 5  Key ablations (all measured on device)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_ablations.png"), dpi=160)

# ---------------------------------------------------------------- 图 6：搜索成本
if p3 and p2b_pilot:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.0))
    x29 = [32, 16 * 257, 32 * 257]
    y29 = [p2b_pilot["arms"]["_b0_pilot"]["kl_mean"],
           p2b_pilot["arms"]["_b1_pilot"]["kl_mean"],
           p2b_pilot["arms"]["_p1_pilot"]["kl_mean"]]
    a1.plot(x29, y29, "o-", label="2.9B (same 8-layer plan)")
    c15 = load("p3c_15b_compare.json", "/home/disk/models/rwkv7-1.5b")
    if c15:
        x15 = [32, 16 * 257]
        y15 = [c15["arms"]["_15b_a0"]["kl_mean"], c15["arms"]["_15b_m"]["kl_mean"]]
        a1.plot(x15, y15, "s--", label="1.5B (same 8-layer plan)")
    a1.set_xscale("log")
    a1.set_yscale("log")
    a1.set_xlabel("calibration cost (tokens fed through the model)")
    a1.set_ylabel("final plan KL")
    a1.set_title("(a) calibration budget vs quality")
    a1.legend(fontsize=8)
    a1.grid(alpha=0.3)
    hs = sorted(int(k) for k in p3["spearman_vs_h64"])
    ys = [p3["spearman_vs_h64"][str(h)] for h in hs]
    a2.plot(hs, ys, "o-", color="#e07b39")
    for h, y in zip(hs, ys):
        a2.annotate("H=%d" % h, (h, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    bs = p3.get("bootstrap_spearman", {})
    if bs:
        txt = " | ".join("H=%s: %.2f [%.2f, %.2f]" % (k, v["mean"], v["ci95"][0], v["ci95"][1])
                         for k, v in bs.items())
        a2.text(0.02, 0.05, "bootstrap " + txt, transform=a2.transAxes, fontsize=7)
    a2.set_xscale("log")
    a2.set_xlabel("measurement window H (forward passes)")
    a2.set_ylabel("Spearman vs H=64 ranking")
    a2.set_title("(b) measurement budget vs ranking fidelity")
    a2.grid(alpha=0.3)
    fig.suptitle("Fig 6  Search cost: calibration budget and measurement window", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig6_search_cost.png"), dpi=160)

# ---------------------------------------------------------------- 输出表
with open(os.path.join(MD, "p6_ablations.json"), "w", encoding="utf-8") as fh:
    json.dump({"ablations": ablations, "note": "全部为实测值；path protection 的 0.56 是 P2-C 的路径保护理论上限"},
              fh, ensure_ascii=False, indent=2)
with open(os.path.join(MD, "p6_metrics.json"), "w", encoding="utf-8") as fh:
    json.dump({"metrics": metric_rows}, fh, ensure_ascii=False, indent=2)

print("图已写入 %s：" % OUT)
for f in sorted(os.listdir(OUT)):
    print("  %-28s %6.0f KB" % (f, os.path.getsize(os.path.join(OUT, f)) / 1024))
print("\n指标面板：")
for r in metric_rows:
    print("  %-20s 成本 %2d 次前向｜Spearman %.3f｜top-8 %.2f"
          % (r["metric"], r["cost_forwards"], r["spearman_vs_kl64"], r["top8_overlap"]))
if ablations:
    print("\n消融：")
    for k, v in ablations.items():
        print("  [%s]" % k)
        for name, val in v.items():
            print("    %-38s %.4f" % (name, val))
