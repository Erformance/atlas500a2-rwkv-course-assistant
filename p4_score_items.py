#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P4 ①：对同一批题目分别用 NPU（打分服务）或 CPU（PyTorch 参考）算 logp。

两种模式必须**分开跑**：NPU 引擎约 5.5GB + CPU bf16 权重约 5.9GB 同时驻留会超过
这台 11.5GB 无 swap 设备的上限，所以先 NPU 后 CPU，各自落盘再比对（`p4_compare_items.py`）。

关键口径：tokenization 用 `tok(ctx) + tok(cont)`（与打分服务一致），
不把 ctx+cont 拼起来整体分词——否则边界处的 BPE 合并会引入不可比差异。

用法：
  python p4_score_items.py --mode npu --base-url http://127.0.0.1:8100 --out p4_npu_scores.json
  /home/disk/miniconda3/envs/rwkv7/bin/python p4_score_items.py --mode cpu --model-dir ... --out p4_cpu_scores.json
"""

import argparse
import json
import time
import urllib.request


def score_npu(items, base_url):
    pairs = []
    for it in items:
        for c in it["choices"]:
            pairs.append([it["ctx"], c])
    req = urllib.request.Request(base_url.rstrip("/") + "/loglikelihood",
                                 data=json.dumps({"requests": pairs}).encode(),
                                 headers={"Content-Type": "application/json"})
    res = json.loads(urllib.request.urlopen(req, timeout=3600).read().decode())["results"]
    out = []
    k = 0
    for it in items:
        row = {"index": it["index"], "label": it["label"], "logp": []}
        for _ in it["choices"]:
            row["logp"].append(round(float(res[k][0]), 6))
            k += 1
        out.append(row)
    return out


def score_cpu(items, model_dir, dtype="bfloat16"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # aarch64 上 MKLDNN 的 bf16 matmul 未实现（mkldnn_matmul 报错），关掉走回退实现；
    # 我们只需要几十个 token 的前向，慢一点没关系。
    try:
        torch.backends.mkldnn.enabled = False
    except Exception:
        pass
    torch.set_num_threads(4)
    dt = {"bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True,
                                                 dtype=dt).eval()
    out = []
    for it in items:
        ctx_ids = tok.encode(it["ctx"])
        row = {"index": it["index"], "label": it["label"], "logp": []}
        for c in it["choices"]:
            cont_ids = tok.encode(c)
            ids = torch.tensor([ctx_ids + cont_ids], dtype=torch.long)
            with torch.no_grad():
                logits = model(ids).logits[0]           # (T, vocab)
            logprobs = torch.log_softmax(logits.float(), dim=-1)
            total = 0.0
            for j, t in enumerate(cont_ids):
                pos = len(ctx_ids) + j - 1              # 预测第 j 个续写 token 的位置
                total += float(logprobs[pos, t])
            row["logp"].append(round(total, 6))
        out.append(row)
        print("  #%d cpu logp %s" % (it["index"], row["logp"]), flush=True)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["npu", "cpu"], required=True)
    parser.add_argument("--items", default="p4_items.json")
    parser.add_argument("--model-dir", default="/home/disk/models/rwkv7-2.9b")
    parser.add_argument("--base-url", default="http://127.0.0.1:8100")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.items, encoding="utf-8") as fh:
        items = json.load(fh)["items"]
    t0 = time.time()
    rows = score_npu(items, args.base_url) if args.mode == "npu" \
        else score_cpu(items, args.model_dir, args.dtype)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"mode": args.mode, "model_dir": args.model_dir,
                   "dtype": args.dtype, "seconds": round(time.time() - t0, 1),
                   "rows": rows}, fh, ensure_ascii=False, indent=2)
    for r in rows:
        print("  #%d logp %s → 选择 %d（label %d）%s"
              % (r["index"], r["logp"], int(r["logp"][0] < r["logp"][1]),
                 r["label"], "✓" if int(r["logp"][0] < r["logp"][1]) == r["label"] else "✗"))
    print("%s 模式用时 %.1fs，已写出 %s" % (args.mode, time.time() - t0, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
