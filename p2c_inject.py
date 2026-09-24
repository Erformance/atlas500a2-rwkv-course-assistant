#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P2-C 第二步：量化模型在"每步注入参考状态"与"自留状态"两种模式下的对比。

- 注入模式：每步执行前把 fp16 参考状态灌入 → 只含**单步输出偏差**
- 自由模式：量化模型保留自己的状态 → 单步偏差 + **状态传播**

两者与 fp16 参考的 KL 之差，即"传播"贡献的部分。

用法: python p2c_inject.py --ref-dir /tmp/p2c_ref --suffix _cp_q
输出: p2c_result.json

内存纪律：参考状态用 mmap 读取（np.load(mmap_mode='r')），绝不整体解压；
曾经的教训是压缩 npz + 反复索引导致设备内存被冲垮、整机假死。
"""

import argparse
import glob
import json
import os
import sys
import time
import resource

import numpy as np

MODEL_DIR = "/home/disk/models/rwkv7-2.9b"
sys.path.insert(0, MODEL_DIR)

from rwkv7_serve2 import Engine                       # noqa: E402


def softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def kl_top1(ref_logits, logits):
    kls, agree = [], []
    for t in range(logits.shape[0]):
        p, q = softmax(ref_logits[t]), softmax(logits[t])
        kls.append(float(np.sum(p * (np.log(p + 1e-12) - np.log(q + 1e-12)))))
        agree.append(bool(logits[t].argmax() == ref_logits[t].argmax()))
    return kls, agree


def run(engine, ids, inject_states):
    engine.reset_state()
    out = []
    for t, tok in enumerate(ids):
        if t > 0 and inject_states is not None:
            engine.import_state(inject_states[t - 1])
        out.append(engine.step(int(tok)).copy())
    return np.stack(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref-dir", default="/tmp/p2c_ref")
    parser.add_argument("--suffix", default="_cp_q")
    parser.add_argument("--out", default=os.path.join(MODEL_DIR, "p2c_result.json"))
    args = parser.parse_args()

    ids = np.load(os.path.join(args.ref_dir, "ids.npy")).tolist()
    ref_logits = np.load(os.path.join(args.ref_dir, "logits.npy"))
    # mmap 方式打开每张量文件：索引零拷贝，不产生解压与整体复制
    state_keys = [os.path.basename(p)[:-4] for p in
                  sorted(glob.glob(os.path.join(args.ref_dir, "*.npy")))
                  if os.path.basename(p) not in ("logits.npy", "ids.npy", "shape.npy")]
    states = {k: np.load(os.path.join(args.ref_dir, k + ".npy"), mmap_mode="r")
              for k in state_keys}
    print("参考状态 %d 张量，mmap 打开（峰值内存已用于 %d MB）"
          % (len(states), resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024))

    def step_state(t):
        return {k: np.asarray(v[t]) for k, v in states.items()}

    engine = Engine(MODEL_DIR, suffix=args.suffix)

    t0 = time.time()
    injected = run(engine, ids, [step_state(t) for t in range(len(ids))])
    t_inj = time.time() - t0
    t0 = time.time()
    free = run(engine, ids, None)
    t_free = time.time() - t0

    kl_inj, ag_inj = kl_top1(ref_logits, injected)
    kl_free, ag_free = kl_top1(ref_logits, free)

    result = {
        "suffix": args.suffix,
        "tokens": len(ids),
        "inject": {"kl_mean": round(float(np.mean(kl_inj)), 4),
                   "kl_first8": round(float(np.mean(kl_inj[:8])), 4),
                   "kl_last8": round(float(np.mean(kl_inj[-8:])), 4),
                   "top1_agreement": round(float(np.mean(ag_inj)), 4)},
        "free": {"kl_mean": round(float(np.mean(kl_free)), 4),
                 "kl_first8": round(float(np.mean(kl_free[:8])), 4),
                 "kl_last8": round(float(np.mean(kl_free[-8:])), 4),
                 "top1_agreement": round(float(np.mean(ag_free)), 4)},
        "per_step_kl_inject": [round(x, 4) for x in kl_inj],
        "per_step_kl_free": [round(x, 4) for x in kl_free],
        "seconds": {"inject": round(t_inj, 1), "free": round(t_free, 1)},
    }
    result["propagation_share"] = round(
        max(0.0, (result["free"]["kl_mean"] - result["inject"]["kl_mean"])
            / max(result["free"]["kl_mean"], 1e-9)), 4)

    print("注入模式（无传播）：KL 均值 %.3f（前 8 步 %.3f / 后 8 步 %.3f），top-1 %.1f%%"
          % (result["inject"]["kl_mean"], result["inject"]["kl_first8"],
             result["inject"]["kl_last8"], result["inject"]["top1_agreement"] * 100))
    print("自由模式（含传播）：KL 均值 %.3f（前 8 步 %.3f / 后 8 步 %.3f），top-1 %.1f%%"
          % (result["free"]["kl_mean"], result["free"]["kl_first8"],
             result["free"]["kl_last8"], result["free"]["top1_agreement"] * 100))
    print("传播贡献占比：%.0f%%" % (result["propagation_share"] * 100))
    with open(args.out, "w") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print("已写出 %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
