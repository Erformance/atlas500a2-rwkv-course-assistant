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
import hashlib
import json
import os
import re
import sys
import time

import numpy as np
from tokenizers import Tokenizer

sys.path.insert(0, "/home/disk/models/rwkv7-2.9b")
from rwkv7_serve2 import Engine  # noqa: E402

MODEL_DIR = "/home/disk/models/rwkv7-2.9b"
STATE_CACHE_DIR = os.path.join(MODEL_DIR, "state_cache")

DEFAULT_SYSTEM = (
    "你是一名大学课程助手，名字叫课程助手。"
    "你只能以课程助手的身份回答，不要自称其他模型或公司。"
    "只回答用户最后提出的那个问题，不要复述用户的话，也不要重复之前已经说过的内容。"
    "回答要准确、简洁、条理清楚。"
)

# 停止符：模型经常在答案末尾自己续写下一轮对话，需要截断。
# 注意：不能在答案正文里也生效——讲提示词工程时正文本身就含 "System:"，
# 早期版本因此把整段回答从位置 0 切掉。所以只有出现在"轮次边界"上
# （标记前是换行或句末标点）才算新一轮开始。
# 边界字符里必须带上反引号：int8 档（top8/top16）答完代码块后，模型会直接续写
# "```User: 下一个问题……"（围栏后面没有换行），re 里漏了 ` 就整段漏出去，
# 表现为"回答尾部多出一段自问自答"。
STOP_MARKERS = ("User:", "Assistant:", "System:", "<|endoftext|>")
BOUNDARY_CHARS = "\n。！？.!?…；;\"'）)】」』`"
# 模型有时会把身份前缀一起写出来，去掉它让回答更干净
ROLE_PREFIX = re.compile(r"^\s*(?:课程助手|助手|AI助手|Assistant)\s*[：:]\s*")
# 流式输出时，把可能正在拼装的停止符回看窗口留在缓冲里，不急着打印
LOOKBACK = max(len(m) for m in STOP_MARKERS) + 1      # 多留一个字符，连边界标点一起吃住


def build_prompt(user_text, system_text):
    return "System: %s\n\nUser: %s\n\nAssistant: <think></think>\n" % (system_text, user_text)


def find_stop(text):
    """找最早的、位于轮次边界上的停止符。返回 (位置, 标记)，没有则 (None, None)。"""
    best_pos, best_marker = None, None
    for marker in STOP_MARKERS:
        start = 0
        while True:
            pos = text.find(marker, start)
            if pos == -1:
                break
            # 位置 0 表示模型在复述提示词而不是开启新一轮，不算停止符
            if pos > 0 and text[pos - 1] in BOUNDARY_CHARS:
                if best_pos is None or pos < best_pos:
                    best_pos, best_marker = pos, marker
                break
            start = pos + 1
    return best_pos, best_marker


def trim(text):
    """截断到最早的停止符。返回 (截断后的文本, 是否命中停止符)"""
    pos, _ = find_stop(text)
    if pos is None:
        return text, False
    return text[:pos].rstrip(), True


def looks_like_echo(answer, question):
    """判断回答是不是只在复述用户的问题（小模型偶发）。"""
    a = re.sub(r"\s+", "", answer or "")
    q = re.sub(r"\s+", "", question or "")
    if not a:
        return True
    if a.startswith(("User:", "用户：", "用户:")):
        return True
    if q and a == q:
        return True
    if q and a.startswith(q) and len(a) <= len(q) + 10:
        return True
    return False


class Chat:
    def __init__(self, engine, tokenizer, system=DEFAULT_SYSTEM,
                 repetition_penalty=1.15, rep_window=256):
        self.engine = engine
        self.tokenizer = tokenizer
        self.system = system
        self.repetition_penalty = repetition_penalty
        self.rep_window = rep_window
        self.history_started = False
        self.turns = 0

    def _penalize(self, logits, seen_ids, penalty=None):
        """对已出现过的 token 降权（含 prompt），抑制复读和复述提示词。"""
        pen = self.repetition_penalty if penalty is None else penalty
        if pen <= 1.0 or not seen_ids:
            return logits
        out = logits.copy()
        for tid in set(seen_ids[-self.rep_window:]):
            if out[tid] > 0:
                out[tid] /= pen
            else:
                out[tid] *= pen
        return out

    def reset(self):
        """开新对话：回到"已读完 system 提示词"的状态（有缓存就直接加载）。"""
        self.prepare_system()
        self.history_started = False
        self.turns = 0

    def prepare_system(self):
        """把 system 提示词跑一遍并缓存状态；之后每次开新对话只花一次加载的时间。"""
        prefix = "System: %s\n\n" % self.system
        key = hashlib.md5(prefix.encode("utf-8")).hexdigest()[:12]
        path = os.path.join(STATE_CACHE_DIR, "sys_%s.npz" % key)
        if os.path.exists(path):
            self.engine.import_state(np.load(path))
            return "cache"
        self.engine.reset_state()
        for tok in self.tokenizer.encode(prefix).ids:
            self.engine.step(int(tok))
        os.makedirs(STATE_CACHE_DIR, exist_ok=True)
        np.savez(path, **self.engine.export_state())
        return "built"

    def build_turn_prompt(self, user_text):
        """system 已经在 state 里了，每轮只喂新一轮，靠复用 state 保持上下文。"""
        self.history_started = True
        return "User: %s\n\nAssistant: <think></think>\n" % user_text

    def _generate(self, logits, ids, max_tokens, temperature, top_k, stream, on_delta,
                  penalty=None):
        """从给定 logits 出发生成一段文本。返回 (text, out_ids, stop_reason, gen_s)。"""
        rng = np.random.default_rng(0)
        out_ids = []
        seen_ids = list(ids)          # prompt 也要参与重复惩罚
        text = ""
        shown = 0          # 流式已打印的字符数（配合回看窗口）
        stop_reason = "max_tokens"
        t1 = time.time()
        for _ in range(max_tokens):
            logits = self._penalize(logits, seen_ids, penalty)
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
            seen_ids.append(nxt)
            if stream:
                # 回看窗口内的内容先攒着，避免把 User: 的前半截打出来
                limit = max(0, len(text) - LOOKBACK)
                if limit > shown:
                    chunk = text[shown:limit]
                    sys.stdout.write(chunk)
                    sys.stdout.flush()
                    if on_delta:
                        on_delta(chunk)
                    shown = limit

            # 复读检测：只在正文里生效。代码块内部（``` 之间）的循环、推导式、
            # 重复的属性名都是正常写法，之前会把代码题误判成复读而掐断。
            in_code = text.count("```") % 2 == 1
            tail = text[-24:]
            if (not in_code and len(out_ids) >= 32
                    and sum(not c.isspace() for c in tail) >= 12
                    and tail in text[:-24]):
                stop_reason = "repeat"
                break

            logits = self.engine.step(nxt)
        gen_s = time.time() - t1

        # 生成结束：把回看窗口里剩下的正文补打出来（否则结尾会丢字）
        if stream and len(text) > shown:
            chunk = text[shown:]
            sys.stdout.write(chunk)
            sys.stdout.flush()
            if on_delta:
                on_delta(chunk)
            shown = len(text)
        return text, out_ids, stop_reason, gen_s

    def ask(self, user_text, max_tokens=128, temperature=0.0, top_k=0, stream=False,
            verbose=False, on_delta=None):
        prompt = self.build_turn_prompt(user_text)
        ids = self.tokenizer.encode(prompt).ids

        t0 = time.time()
        logits = None
        for tok in ids:
            logits = self.engine.step(int(tok))
        prefill_s = time.time() - t0

        prompt_logits = logits.copy()
        # 留一份"刚读完提示词"的状态快照：首答若只是在复述用户的话，可以回滚重试
        snap = self.engine.export_state()

        text, out_ids, stop_reason, gen_s = self._generate(
            prompt_logits, ids, max_tokens, temperature, top_k, stream, on_delta)

        if looks_like_echo(text, user_text):
            # 无论 verbose 与否都记一笔：这条会进 service.log，方便统计触发频率
            print("\n[回退重试] 首答疑似复述用户问题：%r" % text[:30], flush=True)
            self.engine.import_state(snap)
            text2, out_ids2, stop2, gen2 = self._generate(
                prompt_logits, ids, max_tokens, temperature, top_k, stream, on_delta,
                penalty=min(self.repetition_penalty * 1.3, 2.0))
            if text2.strip():
                text, out_ids, stop_reason, gen_s = text2, out_ids2, stop2, gen2


        if verbose:
            print("\n[prefill %d token %.2fs；生成 %d token %.2fs → %.2f tok/s；停止原因 %s]"
                  % (len(ids), prefill_s, len(out_ids), gen_s,
                     len(out_ids) / gen_s if gen_s else 0, stop_reason), flush=True)
        self.turns += 1
        text = ROLE_PREFIX.sub("", text.strip(), count=1)
        return text, dict(prefill_s=prefill_s, gen_s=gen_s,
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
    parser.add_argument("--suffix", default="",
                        help="'_q' 用 int8 层模型（缺失的层自动回退 fp16）")
    parser.add_argument("--head-suffix", default=None,
                        help="输出头单独用别的后缀，例如 _q（层仍按 --suffix）")
    parser.add_argument("--plan-file", default=None,
                        help="逐层混合精度计划 JSON（混合精度实验用）")
    parser.add_argument("--repetition-penalty", type=float, default=1.15,
                        help="重复惩罚系数，1.0 表示关闭")
    parser.add_argument("--rep-window", type=int, default=256,
                        help="重复惩罚回看窗口（token 数）")
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(os.path.join(args.model_dir, "tokenizer.json"))
    plan = None
    if args.plan_file:
        with open(args.plan_file) as fh:
            plan = json.load(fh)
        print("[计划] %s：int8 %d 层，head=%s"
              % (args.plan_file, len(plan.get("layers", [])), plan.get("head")), flush=True)
    engine = Engine(args.model_dir, suffix=args.suffix, head_suffix=args.head_suffix,
                    plan=plan)
    chat = Chat(engine, tokenizer, system=args.system,
                repetition_penalty=args.repetition_penalty, rep_window=args.rep_window)
    t_sys = time.time()
    sys_mode = chat.prepare_system()
    print("[system 状态] %s，用时 %.1f s（预填充只需再跑用户这句话）"
          % ("命中缓存" if sys_mode == "cache" else "首次构建并缓存", time.time() - t_sys),
          flush=True)

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
