#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P0-B：同块重放测试。

做法：在线跑若干步，在最后一步捕获每层的**完整输入（执行前）与输出（执行后）**；
然后为每个被测层**新建一个独立的 OM 实例 + 独立输入/输出缓冲**，把捕获的输入喂进去，
与在线输出逐张量比较。要求误差落在该实现自身重复执行的噪声范围内（本实现是确定性的，
预期为逐位一致）。

用法: python p0b_replay.py --steps 8 --layers 0,1,15,31
输出: p0b_replay.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np

MODEL_DIR = "/home/disk/models/rwkv7-2.9b"
sys.path.insert(0, MODEL_DIR)

import acl                                              # noqa: E402
from tokenizers import Tokenizer                        # noqa: E402
from rwkv7_serve2 import (Engine, OmModel, LAYERS, X_BYTES, LOGITS_BYTES,
                          ACL_H2D, ACL_D2H)             # noqa: E402
from capture_calib_pre import PILOT_TEXT                # noqa: E402


def d2h(ptr, size):
    buf = np.empty(size // 4, np.float32)
    acl.rt.memcpy(buf.ctypes.data, size, ptr, size, ACL_D2H)
    return buf


def h2d(arr, size):
    ptr = acl.rt.malloc(size, 0)[0]
    acl.rt.memcpy(ptr, size, np.ascontiguousarray(arr, np.float32).ctypes.data,
                  size, ACL_H2D)
    return ptr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--layers", default="0,1,15,31")
    parser.add_argument("--out", default=os.path.join(MODEL_DIR, "p0b_replay.json"))
    args = parser.parse_args()
    layer_ids = [int(x) for x in args.layers.split(",") if x.strip() != ""]

    tokenizer = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
    ids = tokenizer.encode(PILOT_TEXT).ids[: args.steps]
    engine = Engine(MODEL_DIR)

    # ---- 在线执行，并在最后一步捕获输入/输出 ----
    captured = {}
    for step, tok in enumerate(ids):
        np.copyto(engine.x_host, engine.embedding[int(tok)])
        acl.rt.memcpy(engine.x_in_dev, X_BYTES, engine.x_host.ctypes.data,
                      X_BYTES, ACL_H2D)
        last = step == len(ids) - 1
        for i, om in enumerate(engine.layers):
            if last:
                captured.setdefault(i, {})["in"] = {
                    n: d2h(om.in_dev[k], om.in_sizes[k])
                    for k, n in enumerate(om.in_names)}
            om.execute()
            if last:
                captured[i]["out"] = {
                    n: d2h(om.out_dev[k], om.out_sizes[k])
                    for k, n in enumerate(om.out_names)}
        engine.head.execute()
        acl.rt.memcpy(engine.logits.ctypes.data, LOGITS_BYTES,
                      engine.logits_dev, LOGITS_BYTES, ACL_D2H)

    # ---- 重放：每层用独立实例与独立缓冲 ----
    report = {"steps": len(ids), "layers": {}}
    print("%-8s %-22s %-14s %-14s %s" % ("layer", "输出张量", "最大绝对差",
                                         "平均绝对差", "逐位一致"))
    all_ok = True
    for i in layer_ids:
        path = os.path.join(MODEL_DIR, "layer%d.om" % i)
        om = OmModel(acl, path)                      # 全新实例
        in_ptrs = [h2d(captured[i]["in"][n], om.in_sizes[k])
                   for k, n in enumerate(om.in_names)]
        out_ptrs = [acl.rt.malloc(s, 0)[0] for s in om.out_sizes]
        om.attach(in_ptrs, out_ptrs)
        om.execute()
        rep = {}
        for k, name in enumerate(om.out_names):
            got = d2h(out_ptrs[k], om.out_sizes[k])
            ref = captured[i]["out"][name]
            diff = np.abs(got - ref)
            rep[name] = {"max_abs": float(diff.max()),
                         "mean_abs": float(diff.mean()),
                         "bitwise_equal": bool(np.array_equal(got, ref))}
            all_ok = all_ok and rep[name]["bitwise_equal"]
            print("%-8s %-22s %-14.3e %-14.3e %s"
                  % ("layer%d" % i, name, rep[name]["max_abs"],
                     rep[name]["mean_abs"],
                     "是" if rep[name]["bitwise_equal"] else "否"))
        report["layers"]["layer%d" % i] = rep
        # 释放
        for p, s in zip(in_ptrs, om.in_sizes):
            acl.rt.free(p)
        for p, s in zip(out_ptrs, om.out_sizes):
            acl.rt.free(p)

    report["all_bitwise_equal"] = all_ok
    report["conclusion"] = ("重放与在线执行逐位一致：输入/输出名称映射与缓冲语义可复现" if all_ok
                            else "重放与在线执行存在差异，需检查名称映射或别名假设")
    with open(args.out, "w") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print("\n%s" % report["conclusion"])
    print("已写出 %s" % args.out)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
