#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RWKV-7 G1j 2.9B 在设备 CPU 上的参考推理。

用途：
  1) 验证模型权重完整、能正常加载；
  2) 给后面的 ATC/pyACL 版本提供文本参考；
  3) 打印 state 结构，供单步 ONNX 导出使用。
"""

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="/home/disk/models/rwkv7-2.9b")
    parser.add_argument("--prompt", default="你好，请用一句话介绍你自己。")
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.path, trust_remote_code=True, dtype=dtype
    )
    model.eval()
    print("模型加载完成，用时 %.1f s，dtype=%s" % (time.time() - t0, dtype), flush=True)

    config = model.config
    print("配置: layers=%s hidden=%s heads=%s head_dim=%s vocab=%s"
          % (config.num_hidden_layers, config.hidden_size, config.num_heads,
             config.head_dim, config.vocab_size), flush=True)

    messages = [{"role": "user", "content": args.prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    print("prompt 长度 %d token" % inputs["input_ids"].shape[1], flush=True)

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.tokens,
            do_sample=False,
            use_cache=True,
        )
    dt = time.time() - t0
    gen = out[0][inputs["input_ids"].shape[1]:]
    print("-" * 60)
    print("生成: %r" % tokenizer.decode(gen, skip_special_tokens=True))
    print("生成 %d token 用时 %.2f s（%.2f token/s，CPU 参考值）"
          % (gen.shape[0], dt, gen.shape[0] / dt))

    # 打印一次单步前向的 state 结构，供导出参考
    with torch.no_grad():
        outs = model(**inputs, use_cache=True)
    cache = outs.past_key_values
    print("-" * 60)
    print("logits shape:", tuple(outs.logits.shape))
    print("cache 类型:", type(cache).__name__, "层数:", len(cache))
    layer0 = cache[0]
    print("第 0 层 state:", [(k, tuple(v.shape), str(v.dtype)) for k, v in layer0.items()]
          if isinstance(layer0, dict) else layer0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
