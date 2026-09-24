#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按协议 §9 的格式汇总所有实验：输出 results.jsonl 与 SUMMARY.md。

输入（设备模型目录下）：manifest.json、ab_logits.json、pareto.json、
                       sensitivity.json、sensitivity_mixed.json、p2c_result.json
输出：results.jsonl（协议字段，缺失填 null 并注明原因）、SUMMARY.md（人读表格）

用法: python make_summary.py
"""

import json
import os

MODEL_DIR = "/home/disk/models/rwkv7-2.9b"


def load(name):
    path = os.path.join(MODEL_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def record(run_id, plan_id, task, sample_id, kl=None, score=None,
           latency_ms=None, prefill_ms=None, decode_tokens=32, prefix_tokens=64):
    """按 §9 的字段组织一条记录；没有的字段填 None 并给原因。"""
    return {
        "run_id": run_id, "commit": COMMIT, "model_hash": MODEL_HASH,
        "om_hash": None, "plan_id": plan_id, "task": task,
        "sample_id": sample_id, "seed": 0,
        "prefix_tokens": prefix_tokens, "decode_tokens": decode_tokens,
        "cache_mode": "off", "latency_ms": latency_ms, "prefill_ms": prefill_ms,
        "ttft_ms": None, "peak_memory_bytes": None,
        "kl": kl, "nll": None, "task_score": score, "stop_reason": None,
        "status": "ok",
        "notes": {
            "om_hash": "未逐个计算（.om 共 30+ 个、单个 100MB+，按需再补）",
            "ttft_ms": "本批实验固定 token 驱动，未测首字延迟",
            "peak_memory_bytes": "设备未采集进程峰值内存",
            "nll": "未做语言建模评估（协议 §6 P4 待做）",
        },
    }


RECORDS = []
COMMIT = ""
MODEL_HASH = ""


def main():
    global COMMIT, MODEL_HASH
    manifest = load("manifest.json") or {}
    COMMIT = manifest.get("git_commit", "")
    MODEL_HASH = manifest.get("tokenizer_hash", "")

    # ---- 1. 三臂对照（ab_logits.json）----
    ab = load("ab_logits.json")
    if ab:
        ref = ab.get("reference", {})
        RECORDS.append(record("ab-fp16", "fp16", "logits_kl", "pilot%d" % ab["tokens"],
                              latency_ms=ref.get("decode_ms_per_token"),
                              prefill_ms=(ref.get("prefill_s") or 0) * 1000))
        for arm, v in ab.get("arms", {}).items():
            RECORDS.append(record("ab%s" % arm, arm, "logits_kl", "pilot%d" % ab["tokens"],
                                  kl=v.get("kl_mean"), score=v.get("top1_agreement"),
                                  latency_ms=v.get("decode_ms_per_token"),
                                  prefill_ms=(v.get("prefill_s") or 0) * 1000))

    # ---- 2. 逐层敏感度（两个分布）----
    for fname, tag in (("sensitivity.json", "pilot64"),
                       ("sensitivity_mixed.json", "mixed64")):
        sens = load(fname)
        if not sens:
            continue
        for name, v in sorted(sens.get("targets", {}).items()):
            RECORDS.append(record("sens-%s-%s" % (fname.split(".")[0], name),
                                  "single:%s" % name, "layer_sensitivity", tag,
                                  kl=v.get("kl_mean"), score=v.get("top1_agreement")))

    # ---- 3. 混合精度帕累托（pareto.json）----
    pareto = load("pareto.json")
    if pareto:
        for name, v in pareto.get("plans", {}).items():
            RECORDS.append(record("pareto-%s" % name, name, "logits_kl", "pilot64",
                                  kl=v.get("kl_mean"), score=v.get("top1_agreement"),
                                  latency_ms=(1000.0 / v["decode_tok_s"]
                                              if v.get("decode_tok_s") else None)))

    # ---- 4. P2-C 状态注入 ----
    p2c = load("p2c_result.json")
    if p2c:
        for mode in ("inject", "free"):
            v = p2c.get(mode, {})
            RECORDS.append(record("p2c-%s" % mode, p2c.get("suffix", ""),
                                  "state_injection:%s" % mode, "pilot%d" % p2c.get("tokens", 0),
                                  kl=v.get("kl_mean"), score=v.get("top1_agreement")))

    # ---- 输出 ----
    with open(os.path.join(MODEL_DIR, "results.jsonl"), "w") as fh:
        for r in RECORDS:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    rows = []
    for r in RECORDS:
        rows.append("| %s | %s | %s | %s | %s | %s |"
                    % (r["run_id"], r["plan_id"], r["task"], r["sample_id"],
                       "-" if r["kl"] is None else "%.4f" % r["kl"],
                       "-" if r["task_score"] is None else "%.1f%%" % (r["task_score"] * 100)))
    md = ["# 实验结果汇总（协议 §9 格式）", "",
          "- commit：`%s`" % COMMIT,
          "- tokenizer_hash：`%s`" % MODEL_HASH[:16] + "…" if MODEL_HASH else "",
          "- 记录数：%d（原始明细见 `results.jsonl`）" % len(RECORDS), "",
          "| run_id | plan | task | sample | KL | top-1 |",
          "| --- | --- | --- | --- | --- | --- |"] + rows
    with open(os.path.join(MODEL_DIR, "SUMMARY.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")

    print("已写出 results.jsonl（%d 条）与 SUMMARY.md" % len(RECORDS))
    print("\n".join(md[:8]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
