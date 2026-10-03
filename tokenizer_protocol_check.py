#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""任务书 Task C 的第 4.3 节：lm-eval tokenization protocol 核查。

背景：我们的打分 backend 用 `tok(ctx) + tok(cont)`（先各自分词再拼接，避免边界 BPE 合并），
而 lm-evaluation-harness 的标准 causal-LM reference（`HFLM._encode_pair`）用的是
**整串分词再按 context 长度切**，并且会把 context 的**尾随空格**挪进 continuation：

```python
n_spaces = len(context) - len(context.rstrip())
if n_spaces > 0:
    continuation = context[-n_spaces:] + continuation
    context = context[:-n_spaces]
whole_enc = tok(context + continuation)
context_enc = tok(context)
continuation_enc = whole_enc[len(context_enc):]
```

本脚本用 40 组边界用例逐条比较两者，输出 `tokenization_protocol_report.json`：
  * ours       = 我们的 backend 口径
  * reference  = 上面那段 lm-eval 标准行为（等价实现）
  * joined     = tok(ctx+cont) 的朴素整串口径（用来看边界合并的绝对影响）

用法（设备上，npu22 环境）：
  python tokenizer_protocol_check.py --out tokenization_protocol_report.json
"""

import argparse
import json
import os
import sys

MODEL_DIR = os.environ.get("RWKV_MODEL_DIR", "/home/disk/models/rwkv7-2.9b")

# 40 组边界用例：覆盖尾空格 / 首空格 / 中英边界 / 标点 / 可能跨边界合并的 BPE / 空 context / 换行
CASES = [
    ("How do I ready a guinea pig cage?", " Provide the guinea pig with bedding."),
    ("How do I ready a guinea pig cage?", "Provide the guinea pig with bedding."),
    ("How do I ready a guinea pig cage? ", "Provide the guinea pig with bedding."),
    ("Question: dresser\nAnswer:", " replace drawer with bobby pin"),
    ("Question: dresser\nAnswer:", "replace drawer with bobby pin"),
    ("The word is 'hel", "lo'"),                       # 跨边界可合并的 BPE
    ("The word is 'hel", "lo world'"),
    ("import numpy as np\n", "def f(x): return x + 1"),
    ("import numpy as np", " def f(x): return x + 1"),
    ("这是什么？", "这是一个测试。"),
    ("这是什么？ ", "这是一个测试。"),
    ("今天天气很好", "，我们去公园吧。"),
    ("今天天气很好，", "我们去公园吧。"),
    ("请解释一下熵的定义", "：熵是……"),
    ("请解释一下熵的定义：", "熵是……"),
    ("", "The quick brown fox"),
    ("The quick brown fox ", ""),
    ("The quick brown fox", ""),
    ("中文English mixed", " text 混排"),
    ("中文English mixed ", "text 混排"),
    ("1 + 1 =", " 2"),
    ("1 + 1 =", "2"),
    ("def fib(n):\n    return n if n < 2 else", " fib(n-1) + fib(n-2)"),
    ("def fib(n):\n    return n if n < 2 else ", "fib(n-1) + fib(n-2)"),
    ("The answer is (a)", " or (b)?"),
    ("The answer is (a) ", "or (b)?"),
    ("他说：“你好”", "，然后走了。"),
    ("他说：“你好”，", "然后走了。"),
    ("A,B,C", ",D,E"),                                  # 标点拼接
    ("A,B,C ", ",D,E"),
    ("3.14", "15926"),                                   # 数字边界
    ("3.14", " 15926"),
    ("https://example.com", "/path"),                    # URL 边界
    ("https://example.com/", "path"),
    ("2026-10-03", "T12:00:00"),
    ("2026-10-03 ", "T12:00:00"),
    ("The cat sat on the", " mat."),
    ("The cat sat on the ", "mat."),
    ("他说", "：好的。"),
    ("他说：", "好的。"),
]


def lm_eval_encode_pair(tok, context, continuation):
    """lm-evaluation-harness（HF backend）的 `_encode_pair` 等价实现。"""
    n_spaces = len(context) - len(context.rstrip())
    if n_spaces > 0:
        continuation = context[-n_spaces:] + continuation
        context = context[:-n_spaces]
    whole_enc = tok.encode(context + continuation).ids
    context_enc = tok.encode(context).ids
    return context_enc, whole_enc[len(context_enc):]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(MODEL_DIR, "tokenization_protocol_report.json"))
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条（冒烟用）")
    args = parser.parse_args()

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
    cases = CASES[: args.limit] if args.limit else CASES

    rows, mismatch = [], 0
    for ctx, cont in cases:
        ours = tok.encode(ctx).ids + tok.encode(cont).ids
        ref_ctx, ref_cont = lm_eval_encode_pair(tok, ctx, cont)
        reference = ref_ctx + ref_cont
        joined = tok.encode(ctx + cont).ids
        same_ours_ref = ours == reference
        mismatch += 0 if same_ours_ref else 1
        rows.append({
            "context": ctx, "continuation": cont,
            "ours_len": len(ours), "reference_len": len(reference), "joined_len": len(joined),
            "ours_equals_reference": same_ours_ref,
            "ours_equals_joined": ours == joined,
            "ours_ids": ours[:24], "reference_ids": reference[:24],
        })

    report = {
        "task": "C.4.3 lm-eval tokenization protocol 核查",
        "date": "2026-10-03",
        "backend_rule": "tok(ctx) + tok(cont)",
        "reference_rule": "lm-eval HFLM._encode_pair（尾随空格挪进 continuation，整串分词后按 context 长度切）",
        "n_cases": len(rows),
        "n_mismatch_ours_vs_reference": mismatch,
        "verdict": ("PASS：我们的 backend 与 lm-eval 标准口径在这 %d 组边界用例上完全一致" % len(rows)
                    if mismatch == 0 else
                    "FAIL：有 %d/%d 组与标准口径不一致，需按任务书 §4.3 处理"
                    "（保留旧结果 → 修正 backend → 先跑小子集看影响 → 再决定是否全量重跑）"
                    % (mismatch, len(rows))),
        "cases": rows,
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print("用例 %d 组，与 lm-eval 标准口径不一致 %d 组" % (len(rows), mismatch))
    for r in rows[:6]:
        print("  %-34s ours=%d ref=%d joined=%d %s"
              % (r["context"][-30:] + "|" + r["continuation"][:12],
                 r["ours_len"], r["reference_len"], r["joined_len"],
                 "OK" if r["ours_equals_reference"] else "**DIFF**"))
    print(report["verdict"])
    print("已写出 %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
