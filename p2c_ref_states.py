#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P2-C 第一步：跑 fp16 参考轨迹，保存"每步之后的完整状态"与 logits。

状态快照 = 下一层看到的输入状态。32 步 × 21.3MB ≈ 683MB（协议第 3 节算过这个量级）。

**存储格式刻意用未压缩的 .npy（每张量一个文件）**：压缩 npz 每次索引都会重新解压
整个数组，第二步读取时会造成大量瞬时分配，在 11.5GB 无 swap 的设备上会把内存冲垮
（教训：见 P2-C 报告）。未压缩 .npy 可以直接 mmap，读取零拷贝。

用法: python p2c_ref_states.py --tokens 32 --out-dir /tmp/p2c_ref
"""

import argparse
import json
import os
import sys
import time

import numpy as np

MODEL_DIR = "/home/disk/models/rwkv7-2.9b"
sys.path.insert(0, MODEL_DIR)

from tokenizers import Tokenizer                      # noqa: E402
from rwkv7_serve2 import Engine                       # noqa: E402
from capture_calib_pre import PILOT_TEXT              # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--out-dir", default="/tmp/p2c_ref")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
    ids = tokenizer.encode(PILOT_TEXT).ids[: args.tokens]
    engine = Engine(MODEL_DIR, suffix=args.suffix)
    engine.reset_state()

    store = {}
    logits_list = []
    t0 = time.time()
    for t, tok in enumerate(ids):
        logits_list.append(engine.step(int(tok)).copy())
        snap = engine.export_state()
        for key, arr in snap.items():
            store.setdefault(key, []).append(arr.copy())
    print("参考轨迹 %d 步完成，用时 %.1f s" % (len(ids), time.time() - t0))

    total = 0
    for k, v in store.items():
        arr = np.stack(v)
        np.save(os.path.join(args.out_dir, "%s.npy" % k), arr)
        total += arr.nbytes
    np.save(os.path.join(args.out_dir, "logits.npy"), np.stack(logits_list))
    np.save(os.path.join(args.out_dir, "ids.npy"), np.array(ids, np.int32))
    np.save(os.path.join(args.out_dir, "shape.npy"),
            np.array([len(ids), len(store)], np.int32))
    size_mb = (total + np.stack(logits_list).nbytes) / 1e6
    print("已写出 %s（%.0f MB，%d 个状态张量 × %d 步，未压缩 .npy 可 mmap）"
          % (args.out_dir, size_mb, len(store), len(ids)))
    with open(os.path.join(MODEL_DIR, "p2c_ref_meta.json"), "w") as fh:
        json.dump({"tokens": len(ids), "suffix": args.suffix,
                   "dir": args.out_dir, "size_mb": round(size_mb, 1)}, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
