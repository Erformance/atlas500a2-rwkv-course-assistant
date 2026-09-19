#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""不依赖 NPU 的流式输出单元测试：验证停止符截断 + 结尾不丢字。"""

import io
import os
import sys
import contextlib

import numpy as np
from tokenizers import Tokenizer

sys.path.insert(0, "/home/disk/models/rwkv7-2.9b")
import rwkv7_chat as C  # noqa: E402

MODEL_DIR = "/home/disk/models/rwkv7-2.9b"
VOCAB = 65536


class FakeEngine:
    """按脚本吐 token：跳过 prefill 的 step 之后，逐步给出脚本里的 token。"""

    def __init__(self, ids, skip):
        self.ids = ids
        self.skip = skip
        self.n = 0

    def reset_state(self):
        self.n = 0

    def step(self, token_id):
        logits = np.full(VOCAB, -1e9, dtype=np.float32)
        idx = self.n - self.skip
        if idx >= 0:
            nxt = self.ids[idx] if idx < len(self.ids) else 0
            logits[nxt] = 100.0
        self.n += 1
        return logits


def run_case(tokenizer, script_text, max_tokens=200):
    ids = tokenizer.encode(script_text).ids
    prompt_len = len(tokenizer.encode(
        "System: sys\n\nUser: 问题\n\nAssistant: <think></think>\n").ids)
    engine = FakeEngine(ids, skip=prompt_len)
    chat = C.Chat(engine, tokenizer, system="sys")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        text, info = chat.ask("问题", max_tokens=max_tokens, stream=True)
    return buf.getvalue(), text, info


def test_multi_turn_prompt():
    """多轮对话：第一轮带 system，之后只喂新一轮（靠 state 保持上下文）。"""
    chat = C.Chat(None, None, system="sys")
    p1 = chat.build_turn_prompt("第一问")
    p2 = chat.build_turn_prompt("第二问")
    ok = (p1.startswith("System: sys") and "第一问" in p1
          and p2.startswith("User: 第二问") and "System:" not in p2)
    print("[%s] 多轮 prompt 构造" % ("PASS" if ok else "FAIL"))
    print("   第1轮: %r" % p1)
    print("   第2轮: %r" % p2)
    return ok


def test_real_corrupt_input():
    """真实案例：Windows 控制台在'小明的'处插入了残缺字节 e6 98。
    忽略坏字节后，其余文字应当完好。"""
    raw = bytes.fromhex(
        "e5b08fe6988ee59ba0e78eb0e5ae9ee58e8be58a9be8bf87e5a4a7e98597e98592"
        "e8b7b3e6a5bcefbc8ce982a3e4bda0e4bbbbe8aea4e4b8bae5b08fe6988ee698"
        "e79a84e6adbbe698afe59ba0e4b8ba")
    text = raw.decode("utf-8", errors="ignore")
    expect = "小明因现实压力过大酗酒跳楼，那你认为小明的死是因为"
    ok = (text == expect)
    print("[%s] Windows 控制台残缺字节还原" % ("PASS" if ok else "FAIL"))
    print("   解码结果: %r" % text)
    if not ok:
        print("   期望    : %r" % expect)
    return ok


def main():
    tokenizer = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))

    cases = [
        ("换行停止符", "答案是四十二。\n\nUser: 下一个问题", "答案是四十二。"),
        ("无换行停止符", "结论如此。Assistant: 换个人", "结论如此。"),
        ("结尾不丢字", "这句话的最后几个字不能被吞掉。\n\nUser: x", "这句话的最后几个字不能被吞掉。"),
    ]
    ok = True
    for name, script, expect in cases:
        streamed, returned, info = run_case(tokenizer, script)
        target = expect if expect is not None else script
        good = streamed == target and returned == target
        ok = ok and good
        print("[%s] %s" % ("PASS" if good else "FAIL", name))
        print("   流式输出: %r" % streamed)
        print("   返回文本: %r" % returned)
        if expect is not None:
            print("   期望(截断): %r" % target)
        print("   停止原因: %s" % info["stop"])
    print("总体:", "PASS" if ok else "FAIL")
    ok = test_multi_turn_prompt() and ok
    ok = test_real_corrupt_input() and ok
    print("总体(含多轮):", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
