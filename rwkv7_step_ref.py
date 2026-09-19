#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RWKV-7 2.9B 单步前向参考：给 NPU 版本做数值对照。

输出：prompt token、logits 的 top-5、state 结构，并保存 npz 供比对。
"""

import argparse
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def describe(obj, prefix="  ", depth=0):
    if depth > 2:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            print(prefix + "%-16s" % k, end="")
            if isinstance(v, torch.Tensor):
                print("tensor", tuple(v.shape), v.dtype)
            else:
                print(type(v).__name__)
    elif isinstance(obj, (list, tuple)):
        print(prefix + "len", len(obj))
        if len(obj):
            describe(obj[0], prefix + "  ", depth + 1)
    else:
        print(prefix + type(obj).__name__, getattr(obj, "shape", ""))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="/home/disk/models/rwkv7-2.9b")
    parser.add_argument("--prompt", default="你好")
    parser.add_argument("--out", default="/home/disk/models/rwkv7-2.9b/ref_step.npz")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.path, trust_remote_code=True, dtype=torch.bfloat16
    )
    model.eval()

    ids = tokenizer(args.prompt, return_tensors="pt")["input_ids"]
    print("prompt=%r ids=%s" % (args.prompt, ids.tolist()), flush=True)

    t0 = time.time()
    with torch.no_grad():
        out = model(ids, use_cache=True)
    dt = time.time() - t0
    print("单步前向用时 %.1f s" % dt, flush=True)

    logits = out.logits[0, -1].float().numpy()
    top = np.argsort(-logits)[:5]
    print("logits top5:")
    for i in top:
        print("   id=%-6d %-12r logit=%.4f" % (int(i), tokenizer.decode([int(i)]), float(logits[i])))

    np.savez(args.out, ids=ids.numpy(), logits=logits)
    print("已保存参考值到", args.out)

    cache = getattr(out, "state", None)
    print("state 类型:", type(cache).__name__)
    try:
        describe(cache)
        layer0 = cache[0]
        print("layer0 内容:")
        describe(layer0)
    except Exception as exc:  # noqa: BLE001
        print("state 结构打印失败:", exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
