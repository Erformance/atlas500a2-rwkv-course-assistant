#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RWKV-7 2.9B 在 Atlas 500 A2 NPU 上的效果演示（按官方 chat 模板）。"""

import os
import sys
import time

import numpy as np
from tokenizers import Tokenizer

sys.path.insert(0, "/home/disk/models/rwkv7-2.9b")
from rwkv7_serve2 import Engine  # noqa: E402

MODEL_DIR = "/home/disk/models/rwkv7-2.9b"

CASES = [
    ("你好", None, 32),
    ("什么是人工智能？", None, 64),
    ("用一句话解释傅里叶变换。", None, 64),
    ("请给我一句关于春天的诗句。", None, 48),
    ("Python 里怎么判断一个数是不是质数？", None, 96),
    ("什么是机器学习？", "你是一名大学课程助教，回答要简洁准确。", 64),
]


def build_prompt(user, system=None):
    text = ""
    if system:
        text += "System: %s\n\n" % system
    text += "User: %s\n\nAssistant: <think></think>\n" % user
    return text


def main():
    tokenizer = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
    engine = Engine(MODEL_DIR)
    print("=" * 70, flush=True)

    tot_tokens = 0
    tot_time = 0.0
    for user, system, n in CASES:
        prompt = build_prompt(user, system)
        ids = tokenizer.encode(prompt).ids
        engine.reset_state()
        t0 = time.time()
        logits = None
        for tok in ids:
            logits = engine.step(int(tok))
        prefill = time.time() - t0

        out_ids = []
        t1 = time.time()
        for _ in range(n):
            nxt = int(np.argmax(logits))
            out_ids.append(nxt)
            logits = engine.step(nxt)
        gen = time.time() - t1
        tot_tokens += n
        tot_time += gen

        print("【用户】%s" % user, flush=True)
        if system:
            print("【系统】%s" % system, flush=True)
        print("【助手】%s" % tokenizer.decode(out_ids).strip(), flush=True)
        print("   prompt %d token / prefill %.2fs ；生成 %d token / %.2fs / %.2f tok/s"
              % (len(ids), prefill, n, gen, n / gen), flush=True)
        print("-" * 70, flush=True)

    print("合计 %d token / %.2fs → 平均 %.2f token/s" % (tot_tokens, tot_time, tot_tokens / tot_time))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
