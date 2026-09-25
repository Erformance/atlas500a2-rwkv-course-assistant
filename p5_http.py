#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P5 真实请求延迟（协议 §7）：走 HTTP 服务，按用户真实用法测 TTFT 与总时长。

与固定工作量微基准分开报告：这里用真实提示词、真实停止条件（内容触发的停止符会生效），
每个档位发 5 次请求，记录：首字延迟（SSE 第一个 delta）、总时长、输出 token 数、停止原因、
以及"前缀缓存命中"（服务端已缓存 system 状态：同一进程内第一次请求 vs 后续请求）。

用法（设备上，服务已在跑）：
  python p5_http.py --requests 5 --out p5_http.json
"""

import argparse
import json
import os
import time
import urllib.parse
import urllib.request

MODEL_DIR = "/home/disk/models/rwkv7-2.9b"
BASE = "http://127.0.0.1:8000"
QUESTIONS = [
    "什么是梯度下降？",
    "解释一下牛顿第一定律。",
    "什么是熵？",
    "光合作用分哪两个阶段？",
    "写一个判断素数的 Python 函数。",
]


def get(path, timeout=300):
    return json.loads(urllib.request.urlopen(BASE + path, timeout=timeout).read().decode())


def post(path, obj, timeout=300):
    data = json.dumps(obj).encode()
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode())


def stream_once(q, tokens=64):
    """走 SSE 流式接口，测首字延迟与总时长。"""
    url = "%s/chat/stream?tokens=%d&q=%s" % (BASE, tokens, urllib.parse.quote(q))
    t0 = time.time()
    ttft = None
    text = ""
    done = None
    with urllib.request.urlopen(url, timeout=300) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            try:
                d = json.loads(line[6:])
            except Exception:
                continue
            if d.get("delta"):
                if ttft is None:
                    ttft = time.time() - t0
                text += d["delta"]
            if d.get("done"):
                done = d
                break
    total = time.time() - t0
    return {"ttft_s": round(ttft, 2) if ttft else None,
            "total_s": round(total, 2),
            "n_tokens": (done or {}).get("n"),
            "stop": (done or {}).get("stop"),
            "server_prefill_s": (done or {}).get("prefill_s"),
            "server_tok_s": (done or {}).get("tok_s"),
            "answer_head": text[:60]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=5)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--out", default=os.path.join(MODEL_DIR, "p5_http.json"))
    args = parser.parse_args()

    tier = get("/tier")
    report = {"tier": tier["current"], "label": tier["label"], "requests": args.requests,
              "tokens_per_request": args.tokens, "runs": []}
    for i in range(args.requests):
        q = QUESTIONS[i % len(QUESTIONS)]
        post("/chat", {"reset": True})
        r = stream_once(q, args.tokens)
        r["question"] = q
        r["warm"] = i > 0
        report["runs"].append(r)
        print("%-28s 首字 %5.1fs｜总 %5.1fs｜%s token｜%s"
              % (q[:14], r["ttft_s"] or -1, r["total_s"], r["n_tokens"], r["stop"]), flush=True)

    warm = [r for r in report["runs"] if r["warm"]]
    if warm:
        report["warm_median"] = {
            "ttft_s": sorted(r["ttft_s"] for r in warm if r["ttft_s"])[len(warm) // 2],
            "total_s": sorted(r["total_s"] for r in warm)[len(warm) // 2],
        }
    with open(args.out, "w") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print("已写出 %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
