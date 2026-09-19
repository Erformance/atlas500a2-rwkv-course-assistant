#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断停止符截断：打印未经截断的原始输出，以及命中标记的位置和被丢弃的内容。

用途：确认 rwkv7_chat.trim() 是"干净截断"还是"误切正文"。
"""

import sys

sys.path.insert(0, "/home/disk/models/rwkv7-2.9b")

from tokenizers import Tokenizer                      # noqa: E402
from rwkv7_serve2 import Engine                       # noqa: E402
from rwkv7_chat import Chat, STOP_MARKERS             # noqa: E402

MODEL_DIR = "/home/disk/models/rwkv7-2.9b"


def main():
    tokenizer = Tokenizer.from_file(MODEL_DIR + "/tokenizer.json")
    engine = Engine(MODEL_DIR)
    chat = Chat(engine, tokenizer)
    chat.prepare_system()

    def raw_generate(question, n=80):
        """和 Chat.ask 相同的生成流程，但不做任何截断。"""
        prompt = "User: %s\n\nAssistant: <think></think>\n" % question
        ids = tokenizer.encode(prompt).ids
        logits = None
        for tok in ids:
            logits = engine.step(int(tok))
        seen = list(ids)
        text = ""
        for _ in range(n):
            logits = chat._penalize(logits, seen)
            nxt = int(logits.argmax())
            text += tokenizer.decode([nxt])
            seen.append(nxt)
            logits = engine.step(nxt)
        return text

    for question in sys.argv[1:] or ["什么是光合作用？", "简单介绍一下牛顿第一定律"]:
        chat.reset()
        raw = raw_generate(question)
        hits = [(raw.find(m), m) for m in STOP_MARKERS if m in raw]
        hits.sort()
        print("=== %s" % question)
        print("  原始长度 %d，命中标记 %s" % (len(raw), hits if hits else "无"))
        print("  原始尾部 50 字: %r" % raw[-50:])
        if hits:
            pos, marker = hits[0]
            discarded = raw[pos + len(marker):]
            print("  截断点 %d，标记 %r，被丢弃 %d 字: %r"
                  % (pos, marker, len(discarded), discarded[:60]))
            print("  截断后尾部 30 字: %r" % raw[:pos][-30:])
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
