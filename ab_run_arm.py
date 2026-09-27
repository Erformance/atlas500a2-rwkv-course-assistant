#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P2：跑单个量化臂，落盘逐步 logits 与速度（每个臂独立进程，避免设备内存累积）。

用法: python ab_run_arm.py --suffix _cp_q --tokens 64 --out arm_cp.npz
      python ab_run_arm.py --suffix ""    --tokens 64 --out arm_fp16.npz
"""

import argparse
import json
import os
import sys
import time

import numpy as np

MODEL_DIR = os.environ.get("RWKV_MODEL_DIR", "/home/disk/models/rwkv7-2.9b")
sys.path.insert(0, MODEL_DIR)

import acl                                            # noqa: E402
from tokenizers import Tokenizer                      # noqa: E402
from rwkv7_serve2 import Engine, WKV_BYTES, LAYERS    # noqa: E402
from capture_calib_pre import PILOT_TEXT, MIXED_TEXT   # noqa: E402


def take_ids(tokenizer, text, tokens):
    """只取前 `tokens` 个 token，按需逐步截断文本。

    `tokenizers` 在本机对这份语料的编码耗时随长度超线性增长（实测 512 字符 0.22 s、
    4096 字符 36 s、12.7 KB 全文 250 s），直接 encode 全文会让每条臂白白多花几分钟。
    BPE 是自左向右的，取足够长的前缀再截前 `tokens` 个 id 与全文结果一致
    （实测 4 KB 前缀与 8 KB 前缀的前 128 个 id 完全相同）。这里从 1 KB 起步，
    不够就翻倍，直到攒够或用到全文。
    """
    n = 1024
    while True:
        ids = tokenizer.encode(text[:n]).ids
        if len(ids) >= tokens or n >= len(text):
            return ids[:tokens]
        n *= 4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", default="")
    parser.add_argument("--plan-json", default="",
                        help='逐层计划，如 \'{"suffix":"_cp_q","layers":[5],"head":false}\'')
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--text", default="pilot",
                        choices=["pilot", "mixed", "p1val", "p1test"],
                        help="pilot=自然问答文本；mixed=代码为主的混合文本；"
                             "p1val/p1test=P1 语料里独立分割的验证/测试文本")
    parser.add_argument("--decode-steps", type=int, default=32)
    parser.add_argument("--out", required=True)
    parser.add_argument("--meta", default="")
    args = parser.parse_args()

    t_stage = time.time()
    tokenizer = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
    print("[计时] 载入 tokenizer %.1fs" % (time.time() - t_stage), flush=True)
    t_stage = time.time()
    if args.text == "mixed":
        text = MIXED_TEXT
    elif args.text in ("p1val", "p1test"):
        import p1_corpus                       # 与校准分割不重叠
        text = p1_corpus.text_for(args.text[2:])
    else:
        text = PILOT_TEXT
    ids = take_ids(tokenizer, text, args.tokens)
    print("[计时] 取文本+分词 %.1fs（%d token）" % (time.time() - t_stage, len(ids)),
          flush=True)

    plan = json.loads(args.plan_json) if args.plan_json else None
    t_stage = time.time()
    engine = Engine(MODEL_DIR, suffix=args.suffix, plan=plan)
    print("[计时] 建引擎 %.1fs" % (time.time() - t_stage), flush=True)
    engine.reset_state()
    logits_seq = []
    t0 = time.time()
    for tok in ids:
        logits_seq.append(engine.step(int(tok)).copy())
    prefill_s = time.time() - t0

    # 状态摘要（每层 wkv 的均值绝对值；只留摘要，不落盘大数组）
    wkv = []
    for i in range(LAYERS):
        buf = np.empty(WKV_BYTES // 4, np.float32)
        acl.rt.memcpy(buf.ctypes.data, WKV_BYTES, engine.wkv[i], WKV_BYTES, 2)
        wkv.append(float(np.abs(buf).mean()))

    tok_fixed = int(ids[-1])
    t1 = time.time()
    for _ in range(args.decode_steps):
        engine.step(tok_fixed)
    decode_s = time.time() - t1

    t_stage = time.time()
    meta = {
        "suffix": args.suffix,
        "plan": plan,
        "text": args.text,
        "tokens": len(ids),
        "n_quant_layers": engine.n_quant,
        "n_fp16_layers": engine.n_plain,
        "prefill_s": round(prefill_s, 3),
        "decode_ms_per_token": round(decode_s / args.decode_steps * 1000, 2),
        "decode_tok_s": round(args.decode_steps / decode_s, 2),
        "wkv_absmean_first_last": [round(wkv[0], 4), round(wkv[-1], 4)],
        "wkv_absmean_mean": round(float(np.mean(wkv)), 4),
    }
    np.savez_compressed(args.out, logits=np.stack(logits_seq),
                        wkv=np.array(wkv, np.float32))
    print("[计时] 落盘 %.1fs" % (time.time() - t_stage), flush=True)
    if args.meta:
        with open(args.meta, "w") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)
    print("臂 %-8s int8 %2d 层 ｜ prefill %.2fs ｜ decode %.2f ms/token（%.2f tok/s）"
          % (args.suffix or "(fp16)", meta["n_quant_layers"], meta["prefill_s"],
             meta["decode_ms_per_token"], meta["decode_tok_s"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
