#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P4 ①：CPU(PyTorch) 与 NPU(OM) 的**逐 token** logp 对照（协议 §6 的"少量逐题结果"）。

为什么不用整题：这台设备 CPU 上 bf16 matmul 走不了 MKLDNN（aarch64 未实现），
关掉后单 token 前向要 20~30 秒，一整道 PIQA 题（前缀+续写约 60 token）要半小时以上；
两档都要跑 200 题就更不可能。所以这里取"同一段前缀 + 少量续写 token"，
**逐 token 比 logp**——比只比一个求和值更严格，CPU 侧只需十几分钟。

用法：
  python p4_token_check.py --mode npu --ctx "..." --cont "..." --out npu.json
  /home/disk/miniconda3/envs/rwkv7/bin/python p4_token_check.py --mode cpu --ctx "..." --cont "..." --out cpu.json
  python p4_token_check.py --compare npu.json cpu.json
"""

import argparse
import json
import time
import urllib.request


def do_npu(ctx, cont, base_url, max_tokens):
    req = urllib.request.Request(base_url.rstrip("/") + "/loglikelihood_detail",
                                 data=json.dumps({"requests": [[ctx, cont]]}).encode(),
                                 headers={"Content-Type": "application/json"})
    res = json.loads(urllib.request.urlopen(req, timeout=3600).read().decode())["results"][0]
    res["logp"] = res["logp"][:max_tokens]
    res["argmax"] = res["argmax"][:max_tokens]
    res["cont_ids"] = res["cont_ids"][:max_tokens]
    return res


def do_cpu(ctx, cont, model_dir, max_tokens, dtype="bfloat16"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        torch.backends.mkldnn.enabled = False       # aarch64 上 MKLDNN 不支持 bf16 matmul
    except Exception:
        pass
    torch.set_num_threads(4)
    dt = {"bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True,
                                                 dtype=dt).eval()
    ctx_ids = tok.encode(ctx)
    cont_ids = tok.encode(cont)[:max_tokens]
    ids = torch.tensor([ctx_ids + cont_ids], dtype=torch.long)
    with torch.no_grad():
        logits = model(ids).logits[0].float()
    logprobs = torch.log_softmax(logits, dim=-1)
    logp, argmax = [], []
    for j, t in enumerate(cont_ids):
        pos = len(ctx_ids) + j - 1
        logp.append(round(float(logprobs[pos, t]), 6))
        argmax.append(int(logits[pos].argmax()))
    return {"sum_logp": round(sum(logp), 6),
            "is_greedy": all(a == t for a, t in zip(argmax, cont_ids)),
            "logp": logp, "argmax": argmax, "n_ctx": len(ctx_ids),
            "cont_ids": [int(t) for t in cont_ids]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["npu", "cpu", "compare"], required=True)
    parser.add_argument("--ctx", default="")
    parser.add_argument("--cont", default="")
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--model-dir", default="/home/disk/models/rwkv7-2.9b")
    parser.add_argument("--base-url", default="http://127.0.0.1:8100")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--out", default=None)
    parser.add_argument("--compare", nargs=2, default=None)
    args = parser.parse_args()

    if args.mode == "compare":
        with open(args.compare[0], encoding="utf-8") as fh:
            a = json.load(fh)
        with open(args.compare[1], encoding="utf-8") as fh:
            b = json.load(fh)
        pair = list(zip(a["logp"], b["logp"]))
        print("token 数 %d｜ctx %d token" % (len(pair), a["n_ctx"]))
        print("%-6s %-12s %-12s %-10s" % ("位置", "NPU logp", "CPU logp", "|Δ|"))
        for i, (x, y) in enumerate(pair):
            print("%-6d %-12.4f %-12.4f %-10.4f" % (i, x, y, abs(x - y)))
        mx = max(abs(x - y) for x, y in pair) if pair else 0.0
        same_argmax = a["argmax"] == b["argmax"]
        print("\n最大 |Δlogp| = %.4f｜argmax 序列一致：%s" % (mx, "是" if same_argmax else "否"))
        if args.out:
            with open(args.out, "w", encoding="utf-8") as fh:
                json.dump({"max_abs_delta": round(mx, 4), "same_argmax": same_argmax,
                           "npu": a, "cpu": b}, fh, ensure_ascii=False, indent=2)
            print("已写出 %s" % args.out)
        return 0

    t0 = time.time()
    res = do_npu(args.ctx, args.cont, args.base_url, args.max_tokens) if args.mode == "npu" \
        else do_cpu(args.ctx, args.cont, args.model_dir, args.max_tokens, args.dtype)
    out = {"mode": args.mode, "ctx": args.ctx, "cont": args.cont,
           "seconds": round(time.time() - t0, 1), **res}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print("%s：%d 个续写 token，logp %s｜用时 %.0fs｜已写出 %s"
          % (args.mode, len(res["logp"]), res["logp"], time.time() - t0, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
