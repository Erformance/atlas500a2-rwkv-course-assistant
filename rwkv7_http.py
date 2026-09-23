#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""课程助手常驻服务：模型只加载一次，之后每个请求直接推理。

启动:  python rwkv7_http.py --port 8000
接口:  POST /chat   {"q": "问题", "tokens": 128, "reset": false}
       GET  /        简易网页（浏览器直接提问）
       GET  /health  健康检查

注意：pyACL 的 context 不是线程安全的，这里用一把锁把请求串行化。
"""

import argparse
import json
import mimetypes
import os
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer

from tokenizers import Tokenizer

MODEL_DIR = "/home/disk/models/rwkv7-2.9b"
sys.path.insert(0, MODEL_DIR)

from rwkv7_serve2 import Engine              # noqa: E402
from rwkv7_chat import Chat                  # noqa: E402

LOCK = threading.Lock()
ENGINE = None
CHAT = None
START_TIME = time.time()
STATS = {"requests": 0, "tokens": 0, "gen_s": 0.0}
BUSY_FILE = os.path.join(MODEL_DIR, "service.busy")
WEB_DIR = os.path.join(MODEL_DIR, "web")
INDEX_HTML = None          # 启动时从 web/index.html 读入；读不到就回退到内联 PAGE
INDEX_MTIME = 0


def load_index(force=False):
    """把 web/index.html 读进内存缓存；文件改动过就自动重读（改页面不用重启服务）。"""
    global INDEX_HTML, INDEX_MTIME
    path = os.path.join(WEB_DIR, "index.html")
    try:
        mtime = os.path.getmtime(path)
        if not force and INDEX_HTML is not None and mtime == INDEX_MTIME:
            return
        with open(path, "r", encoding="utf-8") as fh:
            INDEX_HTML = fh.read()
        INDEX_MTIME = mtime
        print("[网页] 已加载 %s（%d 字节）" % (path, len(INDEX_HTML)), flush=True)
    except Exception as exc:
        INDEX_HTML = PAGE
        INDEX_MTIME = 0
        print("[网页] 读取 %s 失败（%r），回退到内置简易页面" % (path, exc), flush=True)


def mark_busy(on):
    """单线程服务在推理时无法响应健康探测，用这个标记告诉看门狗"在忙，别重启"。"""
    try:
        if on:
            with open(BUSY_FILE, "w") as fh:
                fh.write(str(os.getpid()))
        elif os.path.exists(BUSY_FILE):
            os.remove(BUSY_FILE)
    except Exception:
        pass

PAGE = """<!doctype html><html lang="zh"><meta charset="utf-8">
<title>课程助手</title>
<style>
body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;max-width:760px;margin:40px auto;padding:0 16px;color:#222}
h1{font-size:20px}
textarea{width:100%;height:90px;font-size:16px;padding:8px;box-sizing:border-box}
button{margin-top:8px;padding:8px 18px;font-size:15px;cursor:pointer}
.meta{color:#888;font-size:13px;margin-left:6px}
#ans{white-space:pre-wrap;background:#f6f7f9;border-radius:8px;padding:14px;margin-top:16px;min-height:60px;line-height:1.6}
</style>
<h1>课程助手（Atlas 500 A2 · RWKV-7 2.9B · NPU）</h1>
<textarea id="q" placeholder="输入你的问题，例如：什么是人工智能？"></textarea>
<div><button onclick="ask()">提问</button>
<button onclick="resetChat()" style="background:#eee">新对话</button>
<span class="meta">问题后可加 #数字 指定生成长度，例如 什么是熵#256</span></div>
<div id="ans"></div>
<script>
async function ask(){
  const q=document.getElementById('q').value.trim(); if(!q)return;
  const box=document.getElementById('ans'); box.textContent='';
  const es=new EventSource('/chat/stream?q='+encodeURIComponent(q));
  es.onmessage=(e)=>{
    const d=JSON.parse(e.data);
    if(d.delta){ box.textContent+=d.delta; window.scrollTo(0,document.body.scrollHeight); }
    if(d.done){ box.title='生成 '+d.n+' token，'+d.tok_s.toFixed(2)+' tok/s，预填充 '+d.prefill_s+'s'; es.close(); }
    if(d.error){ box.textContent+='\n[出错] '+d.error; es.close(); }
  };
  es.onerror=()=>{ es.close(); };
}
async function resetChat(){
  await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({reset:true})});
  document.getElementById('ans').textContent='[已开始新对话]';
}
document.getElementById('q').addEventListener('keydown',e=>{ if(e.key==='Enter'&&e.ctrlKey) ask(); });
</script></html>"""


class Handler(BaseHTTPRequestHandler):
    # 必须用 HTTP/1.0：单线程服务器若保持 keep-alive，会被浏览器的空闲长连接堵死，
    # 新请求排队超时（曾经把看门狗也骗得去重启服务）。一条连接只服务一个请求。
    protocol_version = "HTTP/1.0"

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/health"):
            self._send(200, json.dumps(
                {"ok": True, "uptime_s": round(time.time() - START_TIME, 1), **STATS},
                ensure_ascii=False))
        elif self.path.startswith("/chat/stream"):
            self._stream()
        elif self.path in ("/", "/index.html"):
            load_index()
            self._send(200, INDEX_HTML or PAGE, "text/html; charset=utf-8")
        elif self.path.startswith("/static/"):
            self._static(self.path[len("/static/"):])
        elif self.path.startswith("/favicon"):
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def _static(self, rel):
        """从 web/ 目录提供静态资源，防止路径穿越。"""
        base = os.path.realpath(WEB_DIR)
        path = os.path.realpath(os.path.join(base, urllib.parse.unquote(rel)))
        if not path.startswith(base + os.sep) or not os.path.isfile(path):
            self._send(404, json.dumps({"error": "not found"}))
            return
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript",
                                                  "application/json"):
            ctype += "; charset=utf-8"
        try:
            with open(path, "rb") as fh:
                self._send(200, fh.read(), ctype)
        except Exception as exc:
            self._send(500, json.dumps({"error": repr(exc)}))

    def _stream(self):
        """Server-Sent Events 流式回答：首字一出来就推给浏览器。"""
        query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        q = (query.get("q") or [""])[0].strip()
        n = int((query.get("tokens") or [128])[0])
        if "#" in q:
            q, _, tail = q.rpartition("#")
            if tail.strip().isdigit():
                n = int(tail.strip())
        if not q:
            self._send(400, json.dumps({"error": "缺少 q"}))
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        # 先推一段 2KB 的填充注释：部分手机浏览器与中间代理会攒够一定字节才渲染，
        # 不先冲一下就会出现"等很久才开始出字"（桌面浏览器没这个问题）。
        # 以 ":" 开头是 SSE 注释行，客户端会忽略，不影响数据解析。
        try:
            self.wfile.write(b": " + b" " * 2048 + b"\n\n")
            self.wfile.flush()
        except Exception:
            return

        def send(payload):
            self.wfile.write(("data: %s\n\n" % json.dumps(payload, ensure_ascii=False)).encode())
            self.wfile.flush()

        with LOCK:
            mark_busy(True)
            try:
                def on_delta(chunk):
                    send({"delta": chunk})

                text, info = CHAT.ask(q, n, stream=True, on_delta=on_delta)
                STATS["requests"] += 1
                STATS["tokens"] += info["n"]
                STATS["gen_s"] += info["gen_s"]
                # 把服务端截断、清理之后的最终文本一并下发：
                # 流式过程中可能已经打出了"假续写"的碎片，前端应以这里为准重绘。
                send({"done": True, "answer": text, "n": info["n"],
                      "prefill_s": round(info["prefill_s"], 2),
                      "gen_s": round(info["gen_s"], 2),
                      "tok_s": round(info["n"] / info["gen_s"], 2) if info["gen_s"] else 0,
                      "stop": info["stop"], "turn": CHAT.turns})
            except Exception as exc:
                try:
                    send({"error": repr(exc)})
                except Exception:
                    pass
            finally:
                mark_busy(False)

    def do_POST(self):
        if self.path != "/chat":
            self._send(404, json.dumps({"error": "not found"}))
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(length) or b"{}")
        except Exception as exc:
            self._send(400, json.dumps({"error": "bad json: %s" % exc}))
            return

        with LOCK:
            mark_busy(True)
            try:
                if req.get("reset"):
                    CHAT.reset()
                    self._send(200, json.dumps({"ok": True, "reset": True}))
                    return
                q = (req.get("q") or "").strip()
                if not q:
                    self._send(400, json.dumps({"error": "缺少 q"}))
                    return
                n = int(req.get("tokens") or 128)
                if "#" in q:
                    q, _, tail = q.rpartition("#")
                    if tail.strip().isdigit():
                        n = int(tail.strip())
                text, info = CHAT.ask(
                    q, n,
                    temperature=float(req.get("temperature") or 0.0),
                    top_k=int(req.get("top_k") or 0))
                STATS["requests"] += 1
                STATS["tokens"] += info["n"]
                STATS["gen_s"] += info["gen_s"]
                self._send(200, json.dumps({
                    "answer": text, "n": info["n"],
                    "prefill_s": round(info["prefill_s"], 2),
                    "gen_s": round(info["gen_s"], 2),
                    "tok_s": round(info["n"] / info["gen_s"], 2) if info["gen_s"] else 0,
                    "stop": info["stop"], "turn": CHAT.turns}, ensure_ascii=False))
            except Exception as exc:
                self._send(500, json.dumps({"error": repr(exc)}))
            finally:
                mark_busy(False)

    def log_message(self, fmt, *args):
        if not self.path.startswith("/health"):
            sys.stderr.write("[%s] %s\n" % (time.strftime("%H:%M:%S"), self.requestline))


def main():
    global ENGINE, CHAT
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--head-suffix", default=None)
    parser.add_argument("--plan-file", default=None,
                        help="逐层混合精度计划 JSON（如 plan_top16.json）")
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(os.path.join(args.model_dir, "tokenizer.json"))
    plan = None
    if args.plan_file:
        with open(args.plan_file) as fh:
            plan = json.load(fh)
        print("[计划] %s：int8 %d 层，head=%s"
              % (args.plan_file, len(plan.get("layers", [])), plan.get("head")), flush=True)
    ENGINE = Engine(args.model_dir, suffix=args.suffix, head_suffix=args.head_suffix,
                    plan=plan)
    CHAT = Chat(ENGINE, tokenizer, repetition_penalty=args.repetition_penalty)
    load_index()
    mode = CHAT.prepare_system()
    print("[system 状态] %s" % ("命中缓存" if mode == "cache" else "首次构建并缓存"),
          flush=True)
    print("服务已就绪：http://%s:%d/  （Ctrl-C 退出）" % (args.host, args.port), flush=True)
    # 必须单线程：pyACL 的 context 绑定在创建它的线程上，换线程 execute 会报 107002
    HTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    raise SystemExit(main())
