#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""任务书 Task A 的 A2 臂：**Random Trajectory** 校准采集。

与 `capture_calib_p1.py` 的唯一区别是"在轨迹的哪个位置采样"：

| 臂 | 采样位置 | 样本数 |
| --- | --- | --- |
| A1（= 现有 `_b0_q`） | 32 段各取年龄 0 | 32 |
| **A2（本脚本）** | 同一批 16 段，每段随机位置 × 2（seed 固定） | 32 |
| A3（= 现有 `_b1_q`） | 16 段 × {年龄 0, 256} | 32 |
| A4（= 现有 `_p1_q`） | 32 段 × {0, 64, 256}（高预算参照，不宣称等成本） | 96 |

设计要点（为了做**受控**对照）：
  1. 段集合与 A3 完全相同（`--split calib --limit 16`，同一顺序），
     这样 A2 与 A3 的唯一差别就是"位置怎么选"；
  2. 位置从 `[0, min(traj-max, len(ids))-1]` 均匀随机抽（seed 固定并落盘），
     所以年龄分布是真实运行时更可能出现的形状，而不是人为挑的 0/256；
  3. 采样动作与 p1 完全一致：**在层执行之前**读该层输入（不是执行后的状态），
     否则会重犯 P0 定位到的那个时序错误；
  4. 输出成与 p1 相同的 memmap 格式，`pack_calib_p1.py` 可直接打包。

用法（设备上）：
    python capture_calib_a2.py --split calib --limit 16 --per-seg 2 \
        --seed 20261003 --traj-max 512 \
        --out calib_a2_raw --manifest p1_a2_capture_manifest.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np

MODEL_DIR = os.environ.get("RWKV_MODEL_DIR", "/home/disk/models/rwkv7-2.9b")
sys.path.insert(0, MODEL_DIR)

for _p in ("/home/disk/cann80base/ascend-toolkit/latest/python/site-packages",
           "/home/disk/cann80base/ascend-toolkit/8.0.RC1/python/site-packages"):
    if os.path.isdir(os.path.join(_p, "acl")) and _p not in sys.path:
        sys.path.insert(0, _p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="calib")
    parser.add_argument("--limit", type=int, default=16, help="用前 N 段（与 A3 相同）")
    parser.add_argument("--per-seg", type=int, default=2, help="每段抽几个位置")
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--traj-max", type=int, default=512,
                        help="随机位置的上界（不含）；实际取 min(traj-max, len(ids))")
    parser.add_argument("--out", default=os.path.join(MODEL_DIR, "calib_a2_raw"))
    parser.add_argument("--manifest", default=os.path.join(MODEL_DIR, "p1_a2_capture_manifest.json"))
    args = parser.parse_args()

    import acl
    from tokenizers import Tokenizer
    from rwkv7_serve2 import Engine, LAYERS, X_BYTES, LOGITS_BYTES, ACL_H2D, ACL_D2H
    import p1_corpus

    tokenizer = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
    segs = [s for s in p1_corpus.SEGMENTS if s["split"] == args.split][: args.limit]

    rng = np.random.default_rng(args.seed)
    plan, samples = [], []
    total = 0
    for s in segs:
        ids = tokenizer.encode(s["text"]).ids          # 与 p1 一致：整段文本
        hi = min(args.traj_max, len(ids))
        if hi <= 1:
            print("[警告] %s 只有 %d token，跳过" % (s["id"], len(ids)), flush=True)
            continue
        pos = sorted(int(x) for x in rng.choice(hi, size=min(args.per_seg, hi), replace=False))
        plan.append((s, ids, pos))
        total += len(pos)
        for p in pos:
            samples.append({"corpus_id": s["id"], "kind": s["kind"],
                            "position": p, "state_age": p})

    print("A2 Random Trajectory：%d 段 × 每段 %d 个随机位置 = %d 组样本（seed=%d，上界 %d）"
          % (len(plan), args.per_seg, total, args.seed, args.traj_max), flush=True)
    ages = sorted(s["state_age"] for s in samples)
    print("实得状态年龄：%s" % (ages,), flush=True)

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
        engine.reset_state()                            # 每段从零状态开始，位置即"状态年龄"
        hit_set = set(hits)
        for t in range(min(max(hits) + 1, len(ids))):
            tok = int(ids[t])
            np.copyto(engine.x_host, engine.embedding[tok])
            acl.rt.memcpy(engine.x_in_dev, X_BYTES, engine.x_host.ctypes.data,
                          X_BYTES, ACL_H2D)
            here = t in hit_set
            for i, om in enumerate(engine.layers):
                if here:                                # 执行前采样 = 该层真正的输入
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
            if here:
                sample_idx += 1
                print("[%02d/%d] %-16s %-8s %4d token｜位置 %3d｜样本 %d/%d｜%.0f s"
                      % (si + 1, len(plan), seg["id"], seg["kind"], len(ids), t,
                         sample_idx, total, time.time() - t_all), flush=True)
    if sample_idx != total:
        raise SystemExit("实采 %d 组，与计划 %d 组不一致" % (sample_idx, total))

    for m in maps.values():
        m.flush()
        del m
    print("采样完成：%d 组，用时 %.1f s" % (total, time.time() - t_all), flush=True)

    manifest = {
        "arm": "A2 Random Trajectory",
        "corpus": "p1_corpus",
        "split": args.split,
        "split_sha1": p1_corpus.split_hash(args.split),
        "seed": args.seed,
        "traj_max": args.traj_max,
        "per_seg": args.per_seg,
        "n_segments": len(plan),
        "n_samples": total,
        "segment_ids": [s["id"] for s, _, _ in plan],
        "samples": samples,
        "state_age_distribution": {
            "min": int(min(ages)), "max": int(max(ages)),
            "mean": round(float(np.mean(ages)), 1),
            "n_unique": int(len(set(ages))),
            "ages": ages,
        },
        "raw_dir": args.out,
        "note": "与 A3(_b1_q) 使用同一批 16 段，唯一差别是采样位置随机而非固定 {0,256}",
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
