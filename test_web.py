#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""网页版验收脚本（在能访问设备的工作机上运行）。

用法: python test_web.py [http://192.168.31.50:8000]
"""

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://192.168.31.50:8000"
OK = "✓"
BAD = "✗"
fails = []


def check(label, cond, detail=""):
    print(" %s %s %s" % (OK if cond else BAD, label, detail))
    if not cond:
        fails.append(label)


def get(path, timeout=30):
    return urllib.request.urlopen(BASE + path, timeout=timeout)


print("== 1. 首页与静态托管")
try:
    html = get("/").read().decode("utf-8")
    check("GET / 返回页面", len(html) > 3000, "%d 字节" % len(html))
    for marker, name in (('id="msgs"', "消息容器"), ('id="send"', "发送按钮"),
                         ('id="len"', "长度下拉"), ('id="new"', "新对话按钮"),
                         ("EventSource", "流式接口"), ("prefers-color-scheme", "深色适配"),
                         ("viewport", "移动端适配")):
        check(name, marker in html)
except Exception as exc:
    check("GET /", False, repr(exc))

for path, want in (("/favicon.ico", 204), ("/static/../rwkv7_chat.py", 404),
                   ("/static/nope.txt", 404)):
    try:
        code = get(path, 10).status
    except urllib.error.HTTPError as exc:
        code = exc.code
    except Exception as exc:
        code = repr(exc)
    check("%s -> %s" % (path, want), code == want, "实际 %s" % code)

print("== 2. 健康接口")
try:
    health = json.loads(get("/health").read())
    check("health.ok", health.get("ok") is True)
except Exception as exc:
    check("health", False, repr(exc))

print("== 3. 流式问答（首字延迟与状态数据）")
try:
    q = urllib.parse.quote("什么是算法？")
    t0 = time.time()
    first = None
    text = ""
    done = None
    for raw in get("/chat/stream?tokens=60&q=%s" % q, 300):
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data: "):
            continue
        d = json.loads(line[6:])
        if d.get("delta"):
            if first is None:
                first = time.time() - t0
            text += d["delta"]
        if d.get("done"):
            done = d
            break
        if d.get("error"):
            check("流式问答", False, d["error"])
            break
    check("收到完整回答", bool(done and len(text) > 10), "%d 字" % len(text))
    if done:
        check("首字延迟 < 10s", first is not None and first < 10, "%.1fs" % (first or -1))
        check("速度 > 3 tok/s", done["tok_s"] > 3, "%.2f tok/s" % done["tok_s"])
        check("token 数接近设定值", abs(done["n"] - 60) <= 30, "%d" % done["n"])
        print("    回答预览：%s" % text[:60].replace("\n", " "))
except Exception as exc:
    check("流式问答", False, repr(exc))

print("== 4. 代码类问题（应产生 ``` 围栏）")
try:
    q = urllib.parse.quote("写一个判断素数的Python函数")
    text = ""
    for raw in get("/chat/stream?tokens=150&q=%s" % q, 300):
        line = raw.decode("utf-8", "replace").strip()
        if line.startswith("data: "):
            d = json.loads(line[6:])
            if d.get("delta"):
                text += d["delta"]
            if d.get("done"):
                break
    check("回答含代码围栏", "```" in text, "%d 字" % len(text))
except Exception as exc:
    check("代码类问题", False, repr(exc))

print("== 5. 多轮上下文与重置")
try:
    def post(payload):
        req = urllib.request.Request(
            BASE + "/chat", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=300).read())

    post({"reset": True})
    a1 = post({"q": "什么是人工智能？", "tokens": 60})
    a2 = post({"q": "它和机器学习是什么关系？", "tokens": 60})
    check("多轮第二轮有回答", len(a2["answer"]) > 5, "%d 字" % len(a2["answer"]))
    check("第二轮带上文", "学习" in a2["answer"] or "AI" in a2["answer"] or "人工智能" in a2["answer"])
    post({"reset": True})
    a3 = post({"q": "它和机器学习是什么关系？", "tokens": 60})
    check("重置后不再引用上文", a3["answer"][:20] != a2["answer"][:20])
    print("    第1轮：%s" % a1["answer"][:40].replace("\n", " "))
    print("    第2轮：%s" % a2["answer"][:40].replace("\n", " "))
    print("    重置后：%s" % a3["answer"][:40].replace("\n", " "))
except Exception as exc:
    check("多轮上下文", False, repr(exc))

print()
if fails:
    print("未通过 %d 项：%s" % (len(fails), "、".join(fails)))
    sys.exit(1)
print("全部通过")
