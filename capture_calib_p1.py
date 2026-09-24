#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P1：用分离语料 + 分层状态年龄（0/64/256）抓校准数据。

与 capture_calib_pre.py 的三点区别：
  1) 语料换成 p1_corpus（32 段校准语料，中英问答/叙述 + 少量代码，与验证/测试分离）；
  2) 每段先喂真实前缀，把循环状态推到目标"状态年龄"（0/64/256 步）再采该步的层输入，
     这样校准覆盖的是真实长轨迹，而不是只有文本开头那几步；
  3) 样本直接写进未压缩 .npy 的 memmap（每层每张量一个文件），不在内存里堆 96 组样本
     —— 这台机器 11.5GB 无 swap，之前被内存尖峰坑过三次，P1 不能再犯。

用法（设备上）：
    python capture_calib_p1.py --split calib --out calib_p1_raw --ages 0,64,256
输出：
    calib_p1_raw/layerNN_<tensor>.npy   (样本数, 维度) 的 memmap
    p1_capture_manifest.json            语料分割哈希、年龄、样本数、逐张量 absmax
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

MODEL_DIR = "/home/disk/models/rwkv7-2.9b"
sys.path.insert(0, MODEL_DIR)

# 直接跑本脚本时 CANN 环境未必 source 过，pyACL 的 python 包就找不到 → 自己补上
for _p in ("/home/disk/cann80base/ascend-toolkit/latest/python/site-packages",
           "/home/disk/cann80base/ascend-toolkit/8.0.RC1/python/site-packages"):
    if os.path.isdir(os.path.join(_p, "acl")) and _p not in sys.path:
        sys.path.insert(0, _p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="calib",
                        help="用哪个分割抓校准数据（calib / val / test）")
    parser.add_argument("--out", default=os.path.join(MODEL_DIR, "calib_p1_raw"))
    parser.add_argument("--ages", default="0,64,256", help="状态年龄（token 位置）")
    parser.add_argument("--max-samples", type=int, default=96)
    parser.add_argument("--limit", type=int, default=0,
                        help="只采前 N 段（冒烟测试用，0=全部）")
    parser.add_argument("--manifest", default=os.path.join(MODEL_DIR, "p1_capture_manifest.json"))
    args = parser.parse_args()

    ages = [int(x) for x in args.ages.split(",") if x.strip()]
    import acl
    from tokenizers import Tokenizer
    from rwkv7_serve2 import Engine, LAYERS, X_BYTES, LOGITS_BYTES, ACL_H2D, ACL_D2H
    import p1_corpus

    tokenizer = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
    segs = [s for s in p1_corpus.SEGMENTS if s["split"] == args.split]
    if args.limit:
        segs = segs[: args.limit]
    plan = []
    for s in segs:
        ids = tokenizer.encode(s["text"]).ids
        hits = [a for a in ages if a < len(ids)]
        if len(hits) < len(ages):
            print("[警告] %s 只有 %d token，跳过年龄 %s"
                  % (s["id"], len(ids), [a for a in ages if a not in hits]), flush=True)
        if hits:
            plan.append((s, ids, hits))

    total = sum(len(h) for _, _, h in plan)
    if total > args.max_samples:
        raise SystemExit("样本数 %d 超过 --max-samples %d，请减少段数或年龄"
                         % (total, args.max_samples))
    print("分割 %s：%d 段，状态年龄 %s，共 %d 组校准样本"
          % (args.split, len(plan), ages, total), flush=True)

    engine = Engine(MODEL_DIR)
    os.makedirs(args.out, exist_ok=True)
    maps, absmax = {}, {}

    def mmap_for(i, name, size):
        key = (i, name)
        if key not in maps:
            path = os.path.join(args.out, "layer%02d_%s.npy" % (i, name))
            maps[key] = np.lib.format.open_memmap(
                path, mode="w+", dtype=np.float32, shape=(total, size // 4))
        return maps[key]

    sample_idx = 0
    t_all = time.time()
    for si, (seg, ids, hits) in enumerate(plan):
        engine.reset_state()                       # 每段从零状态开始
        hit_set = set(hits)
        n_fed = max(hits) + 1
        for t in range(min(n_fed, len(ids))):
            tok = int(ids[t])
            np.copyto(engine.x_host, engine.embedding[tok])
            acl.rt.memcpy(engine.x_in_dev, X_BYTES, engine.x_host.ctypes.data,
                          X_BYTES, ACL_H2D)
            here = t in hit_set
            for i, om in enumerate(engine.layers):
                if here:                            # 层执行前采样 = 该层真正的输入
                    for idx, name in enumerate(om.in_names):
                        size = om.in_sizes[idx]
                        buf = np.empty(size // 4, np.float32)
                        acl.rt.memcpy(buf.ctypes.data, size, om.in_dev[idx], size, ACL_D2H)
                        mmap_for(i, name, size)[sample_idx] = buf
                        key = (i, name)
                        m = float(np.abs(buf).max())
                        if m > absmax.get(key, 0.0):
                            absmax[key] = m
                om.execute()
            engine.head.execute()
            acl.rt.memcpy(engine.logits.ctypes.data, LOGITS_BYTES,
                          engine.logits_dev, LOGITS_BYTES, ACL_D2H)
            if here:                                # 每个状态年龄占一行样本
                sample_idx += 1
                print("[%02d/%d] %-16s %-8s %4d token｜年龄 %3d｜样本 %d/%d｜%.0f s"
                      % (si + 1, len(plan), seg["id"], seg["kind"], len(ids),
                         t, sample_idx, total, time.time() - t_all), flush=True)
    if sample_idx != total:
        raise SystemExit("实采 %d 组，与计划 %d 组不一致" % (sample_idx, total))

    for m in maps.values():
        m.flush()
        del m
    print("采样完成：%d 组，用时 %.1f s" % (total, time.time() - t_all), flush=True)

    manifest = {
        "corpus": "p1_corpus",
        "split": args.split,
        "split_sha1": p1_corpus.split_hash(args.split),
        "ages": ages,
        "n_segments": len(plan),
        "n_samples": total,
        "segment_ids": [s["id"] for s, _, _ in plan],
        "raw_dir": args.out,
        "layers": {},
    }
    for i in range(LAYERS):
        names = [n for (li, n) in absmax if li == i]
        manifest["layers"]["layer%02d" % i] = {
            "tensors": names,
            "absmax": {n: round(absmax[(i, n)], 4) for n in names},
        }
    with open(args.manifest, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    print("manifest 已写出 %s" % args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
