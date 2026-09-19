#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""导出 RWKV-7 G1 的输出头（ln_out + head）为单步 ONNX。

输入 x (1, 1, hidden)，输出 logits (1, 1, vocab)。
"""

import argparse
import json
import os

import torch
import torch.nn as nn
from safetensors import safe_open


class Head(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = config["hidden_size"]
        self.ln = nn.LayerNorm(
            hidden, eps=config.get("norm_eps", 1e-5), bias=config.get("norm_bias", True)
        )
        self.head = nn.Linear(hidden, config["vocab_size"], bias=False)

    def forward(self, x):
        return self.head(self.ln(x))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/home/disk/models/rwkv7-2.9b")
    parser.add_argument("--out", default="/home/disk/models/rwkv7-2.9b/head.onnx")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    args = parser.parse_args()
    dtype = getattr(torch, args.dtype)

    config = json.load(open(os.path.join(args.model_dir, "config.json")))
    index = json.load(open(os.path.join(args.model_dir, "model.safetensors.index.json")))["weight_map"]
    wanted = {
        key: index[key]
        for key in ("rwkv7.ln_out.weight", "rwkv7.ln_out.bias", "head.weight")
        if key in index
    }
    tensors = {}
    for shard in sorted(set(wanted.values())):
        with safe_open(os.path.join(args.model_dir, shard), framework="pt") as handle:
            for key in handle.keys():
                if key in wanted:
                    tensors[key] = handle.get_tensor(key)

    model = Head(config).to(dtype).eval()
    state_dict = {
        "ln.weight": tensors["rwkv7.ln_out.weight"].to(dtype),
        "ln.bias": tensors["rwkv7.ln_out.bias"].to(dtype),
        "head.weight": tensors["head.weight"].to(dtype),
    }
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("缺失=%s 多余=%s" % (missing, unexpected))

    x = torch.zeros(1, 1, config["hidden_size"], dtype=dtype)
    with torch.no_grad():
        torch.onnx.export(
            model, (x,), args.out,
            input_names=["x"], output_names=["logits"],
            opset_version=14, do_constant_folding=False, dynamo=False,
        )
    print("已保存 %s（%.1f MB）" % (args.out, os.path.getsize(args.out) / 1e6))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
