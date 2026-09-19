#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""命令行提问客户端：直接问已经在跑的常驻服务，不再单独加载一份模型。

用法:
  python ask_http.py "什么是人工智能？"        # 单次提问
  python ask_http.py                          # 交互模式（/quit、/reset、问题#长度）
环境变量: RWKV_URL（默认 http://127.0.0.1:8000）
"""

import json
import os
import sys
import urllib.parse
import urllib.request

BASE = os.environ.get("RWKV_URL", "http://127.0.0.1:8000")


def ask_stream(question, tokens=128):
    url = "%s/chat/stream?tokens=%d&q=%s" % (
        BASE, tokens, urllib.parse.quote(question))
    resp = urllib.request.urlopen(url, timeout=600)
    info = None
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data: "):
            continue
        data = json.loads(line[6:])
        if data.get("delta"):
            sys.stdout.write(data["delta"])
            sys.stdout.flush()
        if data.get("error"):
            print("\n[出错] %s" % data["error"])
            return None
        if data.get("done"):
            info = data
    return info


def reset():
    req = urllib.request.Request(BASE + "/chat", data=json.dumps({"reset": True}).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def main():
    args = sys.argv[1:]
    if args:
        info = ask_stream(args[0], int(args[1]) if len(args) > 1 else 128)
    else:
        print("（服务已常驻，直接提问；/quit 退出 ｜ /reset 新对话 ｜ 问题后可加 #数字）")
        while True:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                continue
            if line == "/quit":
                break
            if line == "/reset":
                reset()
                print("[已开始新对话]")
                continue
            n = 128
            if "#" in line:
                line, _, tail = line.rpartition("#")
                if tail.strip().isdigit():
                    n = int(tail.strip())
            sys.stdout.write("【助手】")
            info = ask_stream(line, n)
            if info:
                print("\n[prefill %.1fs ｜ %d token %.2fs → %.2f tok/s；停止 %s]"
                      % (info["prefill_s"], info["n"], info["gen_s"], info["tok_s"],
                         info["stop"]))
        return 0
    if info:
        print("\n[prefill %.1fs ｜ %d token %.2fs → %.2f tok/s；停止 %s]"
              % (info["prefill_s"], info["n"], info["gen_s"], info["tok_s"], info["stop"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
