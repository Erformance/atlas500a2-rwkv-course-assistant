#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RWKV-7 2.9B 课程助手对话层（NPU / pyACL）。

在 rwkv7_serve2.py 的零拷贝引擎之上加：
  * 固定 system 提示词（避免身份错乱）
  * 官方 chat 模板（User: ... / Assistant: <think></think>）
  * 停止符检测：生成到下一轮 "User:" / "Assistant:" / "System:" 就截断
  * 重复检测：同一片段反复出现时提前停止
  * 流式输出

用法：
  python rwkv7_chat.py --ask "什么是人工智能？" --tokens 96
  python rwkv7_chat.py --serve            # stdin 逐行提问
"""

import argparse
import os
import sys
import time

import numpy as np
from tokenizers import Tokenizer

sys.path.insert(0, "/home/disk/models/rwkv7-2.9b")
from rwkv7_serve2 import Engine  # noqa: E402

MODEL_DIR = "/home/disk/models/rwkv7-2.9b"

DEFAULT_SYSTEM = (
    "你是一名大学课程助手，名字叫课程助手。"
    "你只能以课程助手的身份回答，不要自称其他模型或公司。"
    "回答要准确、简洁、条理清楚。"
)

# 停止符：模型经常在答案末尾自己续写下一轮对话，需要截断
STOP_MARKERS = ("\n\nUser:", "\nUser:", "\n\nAssistant:", "\nAssistant:",
                "\n\nSystem:", "User:", "Assistant:", "System:", "<|endoftext|>")
# 流式输出时，把可能正在拼装的停止符回看窗口留在缓冲里，不急着打印
LOOKBACK = max(len(m) for m in STOP_MARKERS) - 1


def build_prompt(user_text, system_text):
    return "System: %s\n\nUser: %s\n\nAssistant: <think></think>\n" % (system_text, user_text)


def trim(text):
    """截断到最早的停止符。返回 (截断后的文本, 是否命中停止符)"""
    cut = len(text)
    for marker in STOP_MARKERS:
        pos = text.find(marker)
        if pos != -1:
            cut = min(cut, pos)
    return text[:cut], cut != len(text)


class Chat:
    def __init__(self, engine, tokenizer, system=DEFAULT_SYSTEM):
        self.engine = engine
        self.tokenizer = tokenizer
        self.system = system
        self.history_started = False
        self.turns = 0

    def reset(self):
        """开新对话：清空 state。"""
        self.engine.reset_state()
        self.history_started = False
        self.turns = 0

    def build_turn_prompt(self, user_text):
        """第一轮带 system；之后只喂新一轮，靠复用 state 保持上下文（RWKV 的强项）。"""
        if self.history_started:
            return "User: %s\n\nAssistant: <think></think>\n" % user_text
        self.history_started = True
        return "System: %s\n\nUser: %s\n\nAssistant: <think></think>\n" % (self.system, user_text)

    def ask(self, user_text, max_tokens=128, temperature=0.0, top_k=0, stream=False, verbose=False):
        prompt = self.build_turn_prompt(user_text)
        ids = self.tokenizer.encode(prompt).ids

        t0 = time.time()
        logits = None
        for tok in ids:
            logits = self.engine.step(int(tok))
        prefill_s = time.time() - t0

        rng = np.random.default_rng(0)
        out_ids = []
        text = ""
        shown = 0          # 流式已打印的字符数（配合回看窗口）
        stop_reason = "max_tokens"
        t1 = time.time()
        for _ in range(max_tokens):
            if temperature <= 0:
                nxt = int(np.argmax(logits))
            else:
                scaled = logits / temperature
                if top_k > 0:
                    kth = np.sort(scaled)[-top_k]
                    scaled = np.where(scaled < kth, -1e30, scaled)
                probs = np.exp(scaled - scaled.max())
                probs /= probs.sum()
                nxt = int(rng.choice(len(probs), p=probs))
            piece = self.tokenizer.decode([nxt])

            # 命中停止符：截断，不输出
            trimmed, hit = trim(text + piece)
            if hit:
                stop_reason = "stop_marker"
                text = trimmed
                break

            text += piece
            out_ids.append(nxt)
            if stream:
                # 回看窗口内的内容先攒着，避免把 User: 的前半截打出来
                limit = max(0, len(text) - LOOKBACK)
                if limit > shown:
                    sys.stdout.write(text[shown:limit])
                    sys.stdout.flush()
                    shown = limit

            if len(out_ids) >= 24 and text[-16:] in text[:-16]:
                stop_reason = "repeat"
                break

            logits = self.engine.step(nxt)
        gen_s = time.time() - t1

        # 生成结束：把回看窗口里剩下的正文补打出来（否则结尾会丢字）
        if stream and len(text) > shown:
            sys.stdout.write(text[shown:])
            sys.stdout.flush()
            shown = len(text)

        if verbose:
            print("\n[prefill %d token %.2fs；生成 %d token %.2fs → %.2f tok/s；停止原因 %s]"
                  % (len(ids), prefill_s, len(out_ids), gen_s,
                     len(out_ids) / gen_s if gen_s else 0, stop_reason), flush=True)
        self.turns += 1
        return text.strip(), dict(prefill_s=prefill_s, gen_s=gen_s,
                                  n=len(out_ids), stop=stop_reason)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument("--system", default=DEFAULT_SYSTEM)
    parser.add_argument("--ask", help="问一个问题")
    parser.add_argument("--serve", action="store_true", help="stdin 逐行提问")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(os.path.join(args.model_dir, "tokenizer.json"))
    engine = Engine(args.model_dir)
    chat = Chat(engine, tokenizer, system=args.system)

    if args.ask:
        question = args.ask
        n = args.tokens
        if "#" in question:                      # 支持 "问题#数字" 写法
            question, _, tail = question.rpartition("#")
            if tail.strip().isdigit():
                n = int(tail.strip())
        text, info = chat.ask(question, n, args.temperature, args.top_k, verbose=True)
        print("【助手】%s" % text)
        return 0

    if args.serve:
        print("READY", flush=True)
        print("（/quit 退出 ｜ /reset 开始新对话 ｜ 问题后可加 #数字 指定生成长度）", flush=True)
        print("[环境] stdin编码=%s ｜ 文件系统编码=%s ｜ LANG=%s"
              % (sys.stdin.encoding, sys.getfilesystemencoding(),
                 os.environ.get("LANG", "(未设置)")), flush=True)
        while True:
            raw = sys.stdin.buffer.readline()
            if not raw:
                break
            enc = sys.stdin.encoding or "utf-8"
            try:
                line = raw.decode(enc)
            except UnicodeDecodeError as exc:
                # 不改行为、不做猜测：如实打印现场，跳过这一行（会话不崩）
                print("[输入解码失败] stdin编码=%s 出错=%s" % (enc, exc), flush=True)
                print("[原始字节] %s" % raw[:80].hex(" "), flush=True)
                print("[十六进制长度] %d 字节" % len(raw), flush=True)
                continue
            line = line.strip()
            if line == "/quit":
                break
            if line == "/reset":
                chat.reset()
                print("[已重置对话]", flush=True)
                continue
            if not line:
                continue
            n = args.tokens
            if "#" in line:
                line, _, tail = line.rpartition("#")
                if tail.strip().isdigit():
                    n = int(tail.strip())
            try:
                sys.stdout.write("【助手】")
                sys.stdout.flush()
                _, info = chat.ask(line, n, args.temperature, args.top_k,
                                   stream=True, verbose=False)
                print("\n[prefill %.1fs ｜ %d token %.2fs → %.2f tok/s；停止 %s ｜ 第 %d 轮]"
                      % (info["prefill_s"], info["n"], info["gen_s"],
                         info["n"] / info["gen_s"] if info["gen_s"] else 0,
                         info["stop"], chat.turns),
                      flush=True)
            except Exception as exc:  # 单行出错不要让整个会话挂掉
                print("\n[本行处理出错，已跳过] %r" % (exc,), flush=True)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
