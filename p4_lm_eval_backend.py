#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P4：lm-evaluation-harness 的自定义 backend（协议 §6）。

把 harness 的 loglikelihood / generate_until 请求转发给设备上的打分服务
（`p4_score_server.py`），因此评测口径与我们的推理服务完全一致（同一 tokenizer、
同一份 .om、同一套零拷贝引擎）。

用法（在装了 lm-eval 的 venv 里）：
    python p4_run_eval.py --tasks piqa --limit 20 --base-url http://127.0.0.1:8100
或直接用 harness 的 CLI（本模块会被 p4_run_eval.py 导入并注册）：
    python -m p4_run_eval ...

注意：harness 版本必须冻结并记录在报告里（当前 0.4.13）。
"""

import json
import urllib.request

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model


def _post(base_url, path, payload, timeout=3600):
    req = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))["results"]


@register_model("rwkv7_npu")
class RWKV7NPU(LM):
    """通过本地打分服务调用 Atlas 500 A2 上的 RWKV-7 引擎。"""

    def __init__(self, base_url="http://127.0.0.1:8100", batch_size=8, **kwargs):
        super().__init__()
        self.base_url = base_url
        self._batch_size = int(batch_size)
        self._max_length = 2048
        self.max_length = 2048          # harness 的部分 filter 会读这个属性

    # --- 必要接口 ---------------------------------------------------------
    @property
    def batch_size(self):
        return self._batch_size

    def loglikelihood(self, requests, **kwargs):
        pairs = [tuple(r.args) for r in requests]
        out = []
        for i in range(0, len(pairs), self._batch_size):
            chunk = [list(p) for p in pairs[i:i + self._batch_size]]
            res = _post(self.base_url, "/loglikelihood", {"requests": chunk})
            out.extend((float(a), bool(b)) for a, b in res)
        return out

    def generate_until(self, requests, **kwargs):
        payload = []
        for r in requests:
            ctx, gen_kwargs = r.args
            gen_kwargs = gen_kwargs or {}
            payload.append([ctx, {"max_gen_toks": int(gen_kwargs.get("max_gen_toks", 32)),
                                  "until": gen_kwargs.get("until") or []}])
        out = []
        for i in range(0, len(payload), self._batch_size):
            out.extend(_post(self.base_url, "/generate_until",
                             {"requests": payload[i:i + self._batch_size]}))
        return out

    def loglikelihood_rolling(self, requests, **kwargs):
        texts = [[r.args[0]] for r in requests]
        out = []
        for i in range(0, len(texts), self._batch_size):
            res = _post(self.base_url, "/loglikelihood_rolling",
                        {"requests": texts[i:i + self._batch_size]})
            out.extend((float(a), bool(b)) for a, b in res)
        return out
