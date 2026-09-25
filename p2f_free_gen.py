#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P2-F：强制 token 与自由生成下，量化偏差被放大了多少？

协议 §4 的 F 组：同一初始前缀、同一精度计划，比较
  * 强制口径（现有评估方式）：把 fp16 贪心生成的 token 序列原样喂给量化模型，逐步比 logits；
  * 自由口径：量化模型自己贪心生成，与 fp16 自由生成比 —— 一旦选了不同的 token，
    误差通过"输出决策反馈"继续放大。
记录：强制 KL、自由 KL、放大倍数、首个分歧位置、生成文本。

每个候选跑在独立子进程（同一进程反复 Engine() 会因设备内存不释放报 245000）。

用法:
  python p2f_free_gen.py --prefix 64 --tokens 64 --out p2f_result.json
  python p2f_free_gen.py --candidate p1_top8 --out-part parts/p1_top8.json
"""

import argparse
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
    {"name": "p1_full", "plan": "plan_p1_top33.json"},
    {"name": "cp_full", "suffix": "_cp_q"},
]


def softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def kl(a, b):
    p, q = softmax(a), softmax(b)
    return float(np.sum(p * (np.log(p + 1e-12) - np.log(q + 1e-12))))


def free_run(engine, prefix_ids, n_tokens):
    """贪心自由生成；返回 (tokens, logits)。logits[t] = 预测 tokens[t] 的那一步。"""
    engine.reset_state()
    t0 = time.time()
    logits = None
    for tok in prefix_ids:
        logits = engine.step(int(tok))
    prefill_s = time.time() - t0
    logits = logits.copy()
    toks, lgs = [], []
    t1 = time.time()
    for _ in range(n_tokens):
        nxt = int(np.argmax(logits))
        toks.append(nxt)
        lgs.append(logits.copy())
        logits = engine.step(nxt)
    decode_s = time.time() - t1
    return toks, np.stack(lgs), prefill_s, decode_s


def forced_run(engine, prefix_ids, tokens):
    """把给定 token 序列喂进去，返回每一步**预测该 token** 的 logits（与 free_run 对齐）。"""
    engine.reset_state()
    logits = None
    for tok in prefix_ids:
        logits = engine.step(int(tok))
    lgs = []
    for tok in tokens:
        lgs.append(logits.copy())          # 先取预测，再喂 token
        logits = engine.step(int(tok))
    return np.stack(lgs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=int, default=64)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--out", default=os.path.join(MODEL_DIR, "p2f_result.json"))
    parser.add_argument("--candidate", default=None)
    parser.add_argument("--out-part", default=None)
    parser.add_argument("--parts-dir", default=os.path.join(MODEL_DIR, "p2f_parts"))
    args = parser.parse_args()

    from tokenizers import Tokenizer
    from rwkv7_serve2 import Engine
    from capture_calib_pre import PILOT_TEXT

    parts = args.parts_dir
    os.makedirs(parts, exist_ok=True)
    tok_path = os.path.join(parts, "ref_tokens.json")
    logits_path = os.path.join(parts, "ref_logits.npy")
    tokenizer = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
    prefix_ids = tokenizer.encode(PILOT_TEXT).ids[: args.prefix]

    # ---- 子进程模式 ---------------------------------------------------------
    if args.candidate:
        if args.candidate == "fp16":
            t0 = time.time()
            engine = Engine(MODEL_DIR)
            load_s = time.time() - t0
            toks, lgs, prefill_s, decode_s = free_run(engine, prefix_ids, args.tokens)
            np.save(logits_path, lgs)
            with open(tok_path, "w") as fh:
                json.dump(toks, fh)
            entry = {"kind": "reference", "load_s": round(load_s, 1),
                     "prefill_s": round(prefill_s, 2),
                     "decode_tok_s": round(args.tokens / max(decode_s, 1e-6), 2),
                     "text": tokenizer.decode(toks)[:200]}
        else:
            cand = next(c for c in CANDIDATES if c["name"] == args.candidate)
            ref_toks = json.load(open(tok_path))
            ref_lgs = np.load(logits_path)
            if cand.get("plan"):
                plan = json.load(open(os.path.join(MODEL_DIR, cand["plan"])))
                engine = Engine(MODEL_DIR, plan=plan)
            else:
                engine = Engine(MODEL_DIR, suffix=cand.get("suffix", "_p1_q"))
            toks, lgs, prefill_s, decode_s = free_run(engine, prefix_ids, args.tokens)
            flgs = forced_run(engine, prefix_ids, ref_toks)
            div = next((i for i, (a, b) in enumerate(zip(toks, ref_toks)) if a != b), None)
            steps = range(args.tokens)
            kl_forced = float(np.mean([kl(ref_lgs[t], flgs[t]) for t in steps]))
            kl_free = float(np.mean([kl(ref_lgs[t], lgs[t]) for t in steps]))
            # 自由生成要分"分歧前/后"看：分歧前两条轨迹上下文相同，之后是不同上下文之间的比较
            cut = div if div is not None else args.tokens
            kl_free_pre = float(np.mean([kl(ref_lgs[t], lgs[t]) for t in range(cut)])) if cut else None
            post = list(range(cut, args.tokens))
            kl_free_post = float(np.mean([kl(ref_lgs[t], lgs[t]) for t in post])) if post else None
            entry = {"kind": "arm", "n_int8": engine.n_quant,
                     "forced_kl_mean": round(kl_forced, 4),
                     "free_kl_mean": round(kl_free, 4),
                     "free_kl_before_diverge": round(kl_free_pre, 4) if kl_free_pre is not None else None,
                     "free_kl_after_diverge": round(kl_free_post, 4) if kl_free_post is not None else None,
                     "amplify": round(kl_free / max(kl_forced, 1e-9), 3),
                     "diverge_step": div,
                     "prefill_s": round(prefill_s, 2),
                     "decode_tok_s": round(args.tokens / max(decode_s, 1e-6), 2),
                     "text": tokenizer.decode(toks)[:200]}
        with open(args.out_part, "w") as fh:
            json.dump(entry, fh, ensure_ascii=False, indent=2)
        print(json.dumps(entry, ensure_ascii=False)[:400], flush=True)
        return 0

    # ---- 驱动模式 -----------------------------------------------------------
    report = {"prefix_tokens": len(prefix_ids), "decode_tokens": args.tokens, "arms": {}}
    order = [{"name": "fp16"}] + CANDIDATES
    for cand in order:
        part = os.path.join(parts, cand["name"] + ".json")
        if os.path.exists(part) and cand["name"] != "fp16":
            pass
        else:
            cmd = [PY, os.path.join(MODEL_DIR, "p2f_free_gen.py"),
                   "--prefix", str(args.prefix), "--tokens", str(args.tokens),
                   "--candidate", cand["name"], "--out-part", part]
            r = subprocess.run(cmd, capture_output=True, text=True)
            out = (r.stdout or "").strip()
            print("%-8s %s" % (cand["name"], out[-200:] if out else
                               ("失败：%s" % (r.stderr or "")[-200:])), flush=True)
        if os.path.exists(part):
            with open(part) as fh:
                d = json.load(fh)
            if cand["name"] == "fp16":
                report["reference"] = d
            else:
                report["arms"][cand["name"]] = d

    if report["arms"]:
        report["amplify_mean"] = round(
            float(np.mean([v["amplify"] for v in report["arms"].values()])), 3)
        print("\n平均放大倍数 %.2f×（强制→自由）" % report["amplify_mean"], flush=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print("已写出 %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
