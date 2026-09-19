#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""停止符截断的逻辑测试（不加载模型，可直接在设备上跑）。"""

import sys

sys.path.insert(0, "/home/disk/models/rwkv7-2.9b")

from rwkv7_chat import trim, find_stop, looks_like_echo   # noqa: E402

CASES = [
    # (输入, 期望截断结果, 说明)
    ("这是光合作用的过程，释放氧气。User: 请解释呼吸作用", "这是光合作用的过程，释放氧气。",
     "模型在句末另起一轮 → 应截断"),
    ("System: 你是一名大学课程助手，请回答得专业一些。", "System: 你是一名大学课程助手，请回答得专业一些。",
     "正文本身以 System: 开头（复述/举例）→ 不该截断"),
    ("写提示词时，通常写成 System: 你的角色是助教。", "写提示词时，通常写成 System: 你的角色是助教。",
     "正文中间提到 System: 且前面是逗号 → 不该截断"),
    ("比如可以这样写。\nAssistant: 你好，有什么可以帮你？", "比如可以这样写。",
     "换行后的 Assistant: → 应截断"),
    ("这里的 User: 指的是提问方", "这里的 User: 指的是提问方",
     "正文里出现 User: 且前面是空格 → 不该截断"),
    ("答案讲完了。<|endoftext|>", "答案讲完了。",
     "结束符 → 应截断"),
    ("普通的一句话，没有任何标记。", "普通的一句话，没有任何标记。",
     "无标记 → 原样返回"),
]


def main():
    bad = 0
    for text, want, note in CASES:
        got, hit = trim(text)
        ok = got == want
        bad += 0 if ok else 1
        print("%s %s\n    输入: %r\n    期望: %r\n    实际: %r（命中=%s）"
              % ("✓" if ok else "✗", note, text, want, got, hit))

    echo_cases = [
        ("User: 写一段带System提示词的例子", "写一段带System提示词的例子", True, "复述问句"),
        ("", "随便问点什么", True, "空回答视为需要重试"),
        ("熵是一个物理学概念，用于描述系统的无序程度。", "什么是熵？", False, "正常回答"),
        ("什么是熵？", "什么是熵？", True, "原样重复问题"),
    ]
    for answer, question, want, note in echo_cases:
        got = looks_like_echo(answer, question)
        ok = got == want
        bad += 0 if ok else 1
        print("%s 复述判定：%s（答案=%r）" % ("✓" if ok else "✗", note, answer[:24]))

    total = len(CASES) + len(echo_cases)
    print("\n%d/%d 通过" % (total - bad, total))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
