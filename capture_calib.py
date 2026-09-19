#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""抓取 fp16 NPU 流水线里每一层的真实输入，作为 int8 量化校准数据。

label-free 量化给激活值用的是默认刻度，深层激活幅度变化很大时会被截断，
这正是 32 层全 int8 后输出崩坏的主因。这里把真实数据抓下来喂给量化器。

用法: python capture_calib.py --steps 20 --out /home/disk/models/rwkv7-2.9b/calib
产出: calib/layer00.npz ...（每个含该层所有输入在若干 token 上的取值）
"""

import argparse
import os
import sys
import time

import numpy as np

MODEL_DIR = "/home/disk/models/rwkv7-2.9b"
sys.path.insert(0, MODEL_DIR)

# 校准文本要贴近实际用法：中文问答 + 一点公式和代码
TEXT = (
    "System: 你是一名大学课程助手，回答要准确、简洁、条理清楚。\n\n"
    "User: 什么是人工智能？\n\n"
    "Assistant: 人工智能是研究如何让机器表现出智能行为的技术，包括学习、推理和感知。\n\n"
    "User: 请用三句话解释牛顿第二定律。\n\n"
    "Assistant: 物体加速度与合外力成正比，与质量成反比，数学形式是 F = ma。\n\n"
    "User: 写一个判断素数的 Python 函数。\n\n"
    "Assistant: def is_prime(n):\n    if n < 2:\n        return False\n"
    "    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return False\n"
    "    return True\n\n"
    "User: 资本主义社会基本矛盾有哪些表现？\n\n"
    "Assistant: 生产社会化与生产资料私有制之间的矛盾，并表现为周期性的经济危机。\n\n"
    "User: 叔本华的哲学思想是否包含悲观色彩？\n\n"
    "Assistant: 他认为人生本质上是痛苦的，主张通过艺术与自我克制获得解脱。\n\n"
    "User: 为什么历史文物值得保护？\n\n"
    "Assistant: 文物承载着文明的信息，是理解过去、确认身份的重要依据。\n"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--out", default=os.path.join(MODEL_DIR, "calib"))
    args = parser.parse_args()

    import acl
    from tokenizers import Tokenizer
    from rwkv7_serve2 import Engine, LAYERS

    os.makedirs(args.out, exist_ok=True)
    tokenizer = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
    all_ids = tokenizer.encode(TEXT).ids
    ids = all_ids[: args.steps]
    print("校准文本共 %d token，取前 %d 步" % (len(all_ids), len(ids)), flush=True)

    engine = Engine(MODEL_DIR)                      # 不带 suffix = fp16
    store = [{n: [] for n in engine.layers[i].in_names} for i in range(LAYERS)]

    t0 = time.time()
    for tok in ids:
        engine.step(int(tok))
        for i, om in enumerate(engine.layers):
            for name, ptr, size in zip(om.in_names, om.in_dev, om.in_sizes):
                buf = np.empty(size // 4, np.float32)
                acl.rt.memcpy(buf.ctypes.data, size, ptr, size, 2)   # D2H
                store[i][name].append(buf)
    print("抓取 %d 步用时 %.1f s" % (len(ids), time.time() - t0), flush=True)

    stats = []
    for i, data in enumerate(store):
        path = os.path.join(args.out, "layer%02d.npz" % i)
        np.savez_compressed(path, **{k: np.stack(v) for k, v in data.items()})
        x = data.get("x", [np.zeros(1)])[0]
        wkv = data.get("wkv_0", [np.zeros(1)])[0]
        stats.append((i, float(np.abs(x).max()), float(np.abs(wkv).max())))
        print("  layer%02d 写出 %s（x 峰值 %.3f，wkv 峰值 %.3f）"
              % (i, os.path.basename(path), stats[-1][1], stats[-1][2]), flush=True)

    print("\n各层激活幅度（峰值）：")
    for i, xm, wm in stats:
        print("  layer%02d  x=%-10.3f wkv=%-10.3f" % (i, xm, wm))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
