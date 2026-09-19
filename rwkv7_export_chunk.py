#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 RWKV-7 G1 2.9B 按 chunk（若干层）导出成单步 ONNX，供 ATC 编译。

每个 chunk 的输入：
    x           (1, 1, hidden)      当前 token 的隐状态
    v_first     (1, 1, hidden)      跨层传递的 value-residual（第 0 层产出）
    每层三个 state：att_shift (1, hidden)、ffn_shift (1, hidden)、wkv (1, heads, hs, hs) fp32
输出：新的 x、v_first 以及每层更新后的三个 state。

权重直接从 safetensors 按层读取，避免把整模 5.8GB 载入内存。
"""

import argparse
import json
import os
import shutil
import sys

import torch
import torch.nn as nn
from safetensors import safe_open


def ensure_package(model_dir):
    """modeling_rwkv7.py 用的是相对导入，需要把它俩放进一个包目录再 import。"""
    pkg = os.path.join(model_dir, "export_pkg")
    os.makedirs(pkg, exist_ok=True)
    init_file = os.path.join(pkg, "__init__.py")
    if not os.path.exists(init_file):
        open(init_file, "w").close()
    for name in ("modeling_rwkv7.py", "configuration_rwkv7.py"):
        src = os.path.join(model_dir, name)
        dst = os.path.join(pkg, name)
        if not os.path.exists(dst) or os.path.getmtime(src) > os.path.getmtime(dst):
            shutil.copyfile(src, dst)
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
    return pkg


def build_blocks(model_dir, start, count, dtype, seq_len=1):
    ensure_package(model_dir)
    from export_pkg.configuration_rwkv7 import Rwkv7Config
    from export_pkg.modeling_rwkv7 import Rwkv7Block

    config = Rwkv7Config.from_pretrained(model_dir)
    # T=1 用 eager(recurrent) 单步实现；T>1 必须用 chunked（recurrent 会对 t 展开）
    config.wkv_implementation = "eager" if seq_len == 1 else "chunked"

    index = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    want = {}
    for layer in range(start, start + count):
        prefix = "rwkv7.blocks.%d." % layer
        for key, shard in index.items():
            if key.startswith(prefix):
                want[key] = shard

    tensors = {}
    for shard in sorted(set(want.values())):
        with safe_open(os.path.join(model_dir, shard), framework="pt") as handle:
            for key in handle.keys():
                if key in want:
                    tensors[key] = handle.get_tensor(key)

    blocks = nn.ModuleList()
    for layer in range(start, start + count):
        block = Rwkv7Block(config, layer)
        prefix = "rwkv7.blocks.%d." % layer
        state_dict = {k[len(prefix):]: v for k, v in tensors.items() if k.startswith(prefix)}
        missing, unexpected = block.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print("  [warn] layer %d missing=%s unexpected=%s"
                  % (layer, missing[:3], unexpected[:3]))
        blocks.append(block.to(dtype).eval())
        print("  layer %d 载入 %d 个权重张量" % (layer, len(state_dict)))
    return config, blocks


class Chunk(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = blocks

    def forward(self, x, v_first, *states):
        outputs = []
        for i, block in enumerate(self.blocks):
            att_shift = states[3 * i]
            ffn_shift = states[3 * i + 1]
            wkv = states[3 * i + 2]

            hidden = block.ln0(x) if block.layer_id == 0 else x
            attn_out, v_first, att_shift, wkv = block.att(
                block.ln1(hidden), v_first, att_shift, wkv, None, None
            )
            hidden = hidden + attn_out
            ffn_out, ffn_shift = block.ffn(block.ln2(hidden), ffn_shift, None, None)
            hidden = hidden + ffn_out

            outputs.extend([att_shift, ffn_shift, wkv])
        return (hidden, v_first, *outputs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/home/disk/models/rwkv7-2.9b")
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--seq-len", type=int, default=1,
                        help=">1 时导出多 token（prefill）版本，使用 chunked 实现")
    parser.add_argument("--dynamic-t", action="store_true",
                        help="T 轴用动态维度（配合 ATC --dynamic_dims）")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    print("导出层 %d..%d，dtype=%s" % (args.start, args.start + args.layers - 1, args.dtype))
    config, blocks = build_blocks(args.model_dir, args.start, args.layers, dtype, args.seq_len)
    chunk = Chunk(blocks).eval()

    C = config.hidden_size
    H, N = config.num_heads, config.head_dim
    T = 1 if args.dynamic_t else args.seq_len
    x = torch.zeros(1, T, C, dtype=dtype)
    v_first = torch.zeros(1, T, C, dtype=dtype)
    states = []
    for _ in range(args.layers):
        states.append(torch.zeros(1, C, dtype=dtype))
        states.append(torch.zeros(1, C, dtype=dtype))
        states.append(torch.zeros(1, H, N, N, dtype=torch.float32))

    with torch.no_grad():
        traced = chunk(x, v_first, *states)
    print("试跑通过，输出 %d 个张量，第一个 shape=%s" % (len(traced), tuple(traced[0].shape)))

    names_in = ["x", "v_first"]
    names_out = ["x", "v_first"]
    for i in range(args.layers):
        names_in += ["att_shift_%d" % i, "ffn_shift_%d" % i, "wkv_%d" % i]
        names_out += ["att_shift_out_%d" % i, "ffn_shift_out_%d" % i, "wkv_out_%d" % i]

    with torch.no_grad():
        dynamic_axes = {"x": {1: "T"}, "v_first": {1: "T"}} if args.dynamic_t else None
        torch.onnx.export(
            chunk,
            (x, v_first, *states),
            args.out,
            input_names=names_in,
            output_names=names_out,
            dynamic_axes=dynamic_axes,
            opset_version=14,
            do_constant_folding=False,
            dynamo=False,
        )
    print("已保存 %s（%.1f MB）" % (args.out, os.path.getsize(args.out) / 1e6))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
