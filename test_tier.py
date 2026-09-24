#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""档位切换端到端自检（在设备上跑，走 127.0.0.1:8000）。

依次切到 fast / precise / balanced：
  1) POST /tier → 看响应是不是 switching
  2) 轮询 /health 直到档位变成新的（并记录就绪耗时、uptime 确认是新进程）
  3) 用 POST /chat 提一个短问题，记录 tok/s
"""

import json
import time
import urllib.parse
import urllib.request

BASE = "http://127.0.0.1:8000"


def get(path, timeout=30):
    return json.loads(urllib.request.urlopen(BASE + path, timeout=timeout).read().decode())


def post(path, obj, timeout=600):
    data = json.dumps(obj).encode()
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode())


def wait_ready(want, limit=120):
    t0 = time.time()
    while time.time() - t0 < limit:
        time.sleep(3)
        try:
            h = get("/health", timeout=10)
            if h.get("ok") and h.get("tier") == want:
                return time.time() - t0, h
        except Exception:
            pass
    return None, None


def ask(q, tokens=64):
    post("/chat", {"reset": True})          # 每问都从新对话开始
    t0 = time.time()
    d = post("/chat", {"q": q, "tokens": tokens})
    return d, time.time() - t0


def main():
    cur = get("/tier")
    print("起始档位：%s（%s）" % (cur["current"], cur["label"]), flush=True)
    for name in ("fast", "precise", "balanced"):
        r = post("/tier", {"tier": name})
        print("\n→ 切换 %s：响应 %s" % (name, json.dumps(r, ensure_ascii=False)), flush=True)
        dt, h = wait_ready(name)
        if not dt:
            print("  ✗ 等不到 %s 就绪（超时）" % name, flush=True)
            continue
        print("  ✓ 就绪用时 %.1f s｜档位=%s｜新进程 uptime=%.1fs"
              % (dt, h.get("tier_label"), h.get("uptime_s")), flush=True)
        d, wall = ask("什么是熵？", 64)
        if "answer" not in d:
            print("  ✗ 提问失败：%s" % json.dumps(d, ensure_ascii=False), flush=True)
            continue
        print("  ✓ 提问：%s…（%d token，服务端 %.2f tok/s，端到端 %.1f s，停止 %s）"
              % (d["answer"][:26], d["n"], d["tok_s"], wall, d["stop"]), flush=True)
    print("\n自检结束：%s" % json.dumps(get("/tier")["current"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
