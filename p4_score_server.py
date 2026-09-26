#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P4：给 lm-evaluation-harness 用的打分服务（协议 §6）。

协议要求"用固定版本的 lm-evaluation-harness + 自定义 backend 完成同 tokenizer 的
loglikelihood/generation 接口"。这里就是那个 backend 的服务端：在设备上加载**一次**引擎
（和网页服务同一套 Engine / 同一份 .om / 同一个 tokenizer.json），用 HTTP 暴露两个接口。

    POST /loglikelihood   {"requests": [[ctx, cont], ...]}
                          → [[sum_logp, is_greedy], ...]
    POST /generate_until  {"requests": [[ctx, {"max_gen_toks": 32, "until": ["\n\n"]}], ...]}
                          → [text, ...]
    GET  /health

口径说明：
  * ctx 先整体预填充（从零状态开始，不跨请求复用状态）；continuation 的每个 token 用
    "喂完前缀后的那一步 logits"取 log_softmax；
  * is_greedy = continuation 的每个 token 都恰好是当时的 argmax（harness 用于判定
    "答案是否被截断成前缀"）；
  * 服务是单线程（pyACL context 绑线程），请求串行处理。

用法（设备上，独占 NPU 时）：
    python p4_score_server.py --port 8100 [--plan-file plan_p1_top8.json | --suffix _p1_q]
"""

import argparse
import json
import math
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

MODEL_DIR = "/home/disk/models/rwkv7-2.9b"
sys.path.insert(0, MODEL_DIR)
# 直接跑本脚本时 CANN 环境未必 source 过，pyACL 的 python 包就找不到 → 自己补上
for _p in ("/home/disk/cann80base/ascend-toolkit/latest/python/site-packages",
           "/home/disk/cann80base/ascend-toolkit/8.0.RC1/python/site-packages"):
    if os.path.isdir(os.path.join(_p, "acl")) and _p not in sys.path:
        sys.path.insert(0, _p)

ENGINE = None
TOKENIZER = None
TIER = {"suffix": "", "plan": None, "name": "fp16"}
STATS = {"loglikelihood_calls": 0, "loglikelihood_pairs": 0, "generate_calls": 0,
         "generated_tokens": 0, "prefill_tokens": 0, "seconds": 0.0}


def log_softmax(x):
    m = float(x.max())
    s = float(sum(math.exp(float(v) - m) for v in x))
    return x - m - math.log(s)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def _send(self, code, obj):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        sys.stderr.write("[%s] %s\n" % (time.strftime("%H:%M:%S"), self.requestline))

    def do_GET(self):
        if self.path.startswith("/health"):
            self._send(200, {"ok": True, "tier": TIER["name"], **STATS})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(length) or b"{}")
        except Exception as exc:
            self._send(400, {"error": "bad json: %s" % exc})
            return
        if self.path.startswith("/loglikelihood"):
            self._send(200, {"results": self._loglikelihood(req.get("requests", []))})
        elif self.path.startswith("/loglikelihood_rolling"):
            self._send(200, {"results": self._rolling(req.get("requests", []))})
        elif self.path.startswith("/generate_until"):
            self._send(200, {"results": self._generate(req.get("requests", []))})
        else:
            self._send(404, {"error": "not found"})

    # ------------------------------------------------------------------ 打分
    def _loglikelihood(self, requests):
        t0 = time.time()
        out = []
        for ctx, cont in requests:
            ids_ctx = TOKENIZER.encode(ctx).ids
            ids_cont = TOKENIZER.encode(cont).ids
            ENGINE.reset_state()
            logits = None
            for tok in ids_ctx:
                logits = ENGINE.step(int(tok))
            if not ids_cont:
                out.append([0.0, False])
                continue
            total, greedy = 0.0, True
            for tok in ids_cont:
                lp = log_softmax(logits)
                total += float(lp[int(tok)])
                if int(logits.argmax()) != int(tok):
                    greedy = False
                logits = ENGINE.step(int(tok))
            out.append([round(total, 6), bool(greedy)])
            STATS["loglikelihood_pairs"] += 1
            STATS["prefill_tokens"] += len(ids_ctx) + len(ids_cont)
        STATS["loglikelihood_calls"] += 1
        STATS["seconds"] += time.time() - t0
        return out

    def _generate(self, requests):
        t0 = time.time()
        out = []
        for ctx, gen_kwargs in requests:
            max_new = int((gen_kwargs or {}).get("max_gen_toks", 32))
            until = (gen_kwargs or {}).get("until") or []
            if isinstance(until, str):
                until = [until]
            ENGINE.reset_state()
            logits = None
            for tok in TOKENIZER.encode(ctx).ids:
                logits = ENGINE.step(int(tok))
            text, n = "", 0
            for _ in range(max_new):
                nxt = int(logits.argmax())
                piece = TOKENIZER.decode([nxt])
                text += piece
                n += 1
                hit = next((u for u in until if u and u in text), None)
                if hit is not None:
                    text = text.split(hit)[0]
                    break
                logits = ENGINE.step(nxt)
            out.append(text)
            STATS["generated_tokens"] += n
        STATS["generate_calls"] += 1
        STATS["seconds"] += time.time() - t0
        return out

    def _rolling(self, requests):
        """整段文本的逐 token log 概率之和（LAMBADA/ppl 类任务用）。"""
        t0 = time.time()
        out = []
        for (text,) in requests:
            ids = TOKENIZER.encode(text).ids
            if len(ids) < 2:
                out.append([0.0, False])
                continue
            ENGINE.reset_state()
            logits = ENGINE.step(int(ids[0]))
            total, greedy = 0.0, True
            for tok in ids[1:]:
                lp = log_softmax(logits)
                total += float(lp[int(tok)])
                if int(logits.argmax()) != int(tok):
                    greedy = False
                logits = ENGINE.step(int(tok))
            out.append([round(total, 6), bool(greedy)])
            STATS["prefill_tokens"] += len(ids)
        STATS["seconds"] += time.time() - t0
        return out


def main():
    global ENGINE, TOKENIZER
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--suffix", default="")
    parser.add_argument("--plan-file", default=None)
    args = parser.parse_args()

    import acl                                        # noqa: F401
    from tokenizers import Tokenizer
    from rwkv7_serve2 import Engine

    plan = None
    if args.plan_file:
        with open(os.path.join(MODEL_DIR, args.plan_file)) as fh:
            plan = json.load(fh)
        TIER.update(plan=args.plan_file, suffix=plan.get("suffix", ""),
                    name="%s(%d 层 int8)" % (args.plan_file, len(plan.get("layers", []))))
    else:
        TIER.update(suffix=args.suffix, name=("fp16" if not args.suffix else args.suffix))

    TOKENIZER = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
    t0 = time.time()
    ENGINE = Engine(MODEL_DIR, suffix=args.suffix, plan=plan)
    print("[打分服务] 引擎加载 %.1fs，档位 %s" % (time.time() - t0, TIER["name"]), flush=True)
    print("服务已就绪：http://%s:%d/" % (args.host, args.port), flush=True)
    HTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    raise SystemExit(main())
