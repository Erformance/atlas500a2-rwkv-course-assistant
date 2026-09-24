#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""课程助手常驻服务：模型只加载一次，之后每个请求直接推理。

启动:  python rwkv7_http.py --port 8000
接口:  POST /chat   {"q": "问题", "tokens": 128, "reset": false}
       GET  /        简易网页（浏览器直接提问）
       GET  /health  健康检查
       GET  /tier    当前推理档位与可选档位表
       POST /tier    {"tier": "fast"} 切档（写 service.tier 后以退出码 75 退出，
                     由 service_supervisor.sh 用新档位重启，约 45 秒）

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

# ---- 推理档位（逐层混合精度）--------------------------------------------
# tiers.json 是档位的唯一事实来源；plan 为空表示全部 fp16。
# 换档要重新加载 .om（约 45 秒），进程自己没法换，所以约定：服务以退出码 75
# 退出，监督脚本 service_supervisor.sh 读到 75 就用新档位重启。
TIERS_FILE = os.path.join(MODEL_DIR, "tiers.json")
TIER_FILE = os.path.join(MODEL_DIR, "service.tier")
SWITCH_EXIT = 75
DEFAULT_TIER = "balanced"
# tiers.json 缺失时的兜底表（设备上正常存在，速度是实测值）
FALLBACK_TIERS = {
    "precise": {"label": "精确档", "desc": "全部 fp16，最稳",
                "plan": "", "tok_s": 5.0},
    "balanced": {"label": "均衡档", "desc": "8 层 int8（KL 0.019，top1 93.8%）",
                 "plan": "plan_top8.json", "tok_s": 5.8},
    "fast": {"label": "快速档", "desc": "16 层 int8（KL 0.049，top1 92.2%）",
             "plan": "plan_top16.json", "tok_s": 7.0},
}
TIERS = {}
CURRENT_TIER = DEFAULT_TIER
TIER_LABEL = ""


def load_tiers():
    """读 tiers.json；读不到就用内置兜底表，保证服务不会因此起不来。"""
    global TIERS
    try:
        with open(TIERS_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict) or not data:
            raise ValueError("内容不是非空对象")
        TIERS = data
    except Exception as exc:
        TIERS = FALLBACK_TIERS
        print("[档位] 读取 %s 失败（%r），改用内置档位表" % (TIERS_FILE, exc), flush=True)


def read_tier_file(default=DEFAULT_TIER):
    try:
        with open(TIER_FILE, "r", encoding="utf-8") as fh:
            name = fh.read().strip()
        if name:
            return name
    except OSError:
        pass
    return default


def write_tier_file(name):
    with open(TIER_FILE, "w", encoding="utf-8") as fh:
        fh.write(name + "\n")


def tier_plan(name):
    """返回 (plan 字典或 None, 计划文件路径或 None)。plan 为空 = 全部 fp16。"""
    rel = ((TIERS.get(name) or {}).get("plan") or "").strip()
    if not rel:
        return None, None
    path = rel if os.path.isabs(rel) else os.path.join(MODEL_DIR, rel)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh), path


def tiers_payload():
    return [{"key": key, "label": item.get("label", key),
             "desc": item.get("desc", ""), "tok_s": item.get("tok_s"),
             "current": key == CURRENT_TIER}
            for key, item in TIERS.items()]


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
                {"ok": True, "uptime_s": round(time.time() - START_TIME, 1),
                 "tier": CURRENT_TIER, "tier_label": TIER_LABEL, **STATS},
                ensure_ascii=False))
        elif self.path.startswith("/tier"):
            self._send(200, json.dumps({"current": CURRENT_TIER, "label": TIER_LABEL,
                                        "tiers": tiers_payload()},
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
        if self.path == "/tier":
            length = int(self.headers.get("Content-Length", 0))
            try:
                req = json.loads(self.rfile.read(length) or b"{}")
            except Exception as exc:
                self._send(400, json.dumps({"error": "bad json: %s" % exc}))
                return
            # 拿锁：生成中途切档要等这一问答完（否则会打断用户正在看的回答）
            with LOCK:
                self._switch_tier((req.get("tier") or "").strip())
            return
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

    def _switch_tier(self, name):
        """切档：写 service.tier，回响应，然后以退出码 75 退出（监督脚本换档重启）。"""
        wanted = TIERS.get(name)
        if not wanted:
            self._send(400, json.dumps(
                {"error": "未知档位 %r（可选：%s）" % (name, "、".join(TIERS))},
                ensure_ascii=False))
            return
        if name == CURRENT_TIER:
            self._send(200, json.dumps({"ok": True, "tier": name, "switching": False},
                                       ensure_ascii=False))
            return
        write_tier_file(name)
        label = wanted.get("label", name)
        print("[档位] 收到切换请求 → %s（%s），退出并让监督脚本换档重启" % (name, label),
              flush=True)
        self._send(200, json.dumps({"ok": True, "tier": name, "label": label,
                                    "switching": True, "eta_s": 60},
                                   ensure_ascii=False))
        # 先让响应真的发出去，再退出；os._exit 不做清理，等价于之前 kill -9 的重启路径
        threading.Timer(1.0, lambda: os._exit(SWITCH_EXIT)).start()

    def log_message(self, fmt, *args):
        if not self.path.startswith("/health"):
            sys.stderr.write("[%s] %s\n" % (time.strftime("%H:%M:%S"), self.requestline))


def main():
    global ENGINE, CHAT, CURRENT_TIER, TIER_LABEL
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--head-suffix", default=None)
    parser.add_argument("--tier", default=None,
                        help="强制指定档位（precise/balanced/fast），默认读 service.tier")
    parser.add_argument("--plan-file", default=None,
                        help="直接指定计划 JSON，覆盖档位（实验用）")
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(os.path.join(args.model_dir, "tokenizer.json"))

    load_tiers()
    plan = None
    if args.plan_file:
        with open(args.plan_file, "r", encoding="utf-8") as fh:
            plan = json.load(fh)
        CURRENT_TIER = "custom"
        TIER_LABEL = "自定义（实验）"
        print("[档位] 直接指定计划 %s：int8 %d 层，head=%s"
              % (args.plan_file, len(plan.get("layers", [])), plan.get("head")), flush=True)
    else:
        name = args.tier or read_tier_file()
        if name not in TIERS:
            print("[档位] 未知档位 %r，回退到 %s" % (name, DEFAULT_TIER), flush=True)
            name = DEFAULT_TIER
        CURRENT_TIER = name
        item = TIERS.get(name) or {}
        TIER_LABEL = item.get("label", name)
        plan, path = tier_plan(name)
        if plan is None:
            print("[档位] %s（%s）｜计划：全部 fp16" % (name, TIER_LABEL), flush=True)
        else:
            print("[档位] %s（%s）｜计划 %s：int8 %d 层，head=%s"
                  % (name, TIER_LABEL, path, len(plan.get("layers", [])),
                     plan.get("head")), flush=True)

    ENGINE = Engine(args.model_dir, suffix=args.suffix, head_suffix=args.head_suffix,
                    plan=plan)
    CHAT = Chat(ENGINE, tokenizer, repetition_penalty=args.repetition_penalty)
    load_index()
    mode = CHAT.prepare_system()
    print("[system 状态] %s" % ("命中缓存" if mode == "cache" else "首次构建并缓存"),
          flush=True)
    print("服务已就绪：http://%s:%d/  档位=%s  （Ctrl+C 退出）"
          % (args.host, args.port, TIER_LABEL or CURRENT_TIER), flush=True)
    # 必须单线程：pyACL 的 context 绑定在创建它的线程上，换线程 execute 会报 107002
    HTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    raise SystemExit(main())
