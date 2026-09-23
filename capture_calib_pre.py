#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P0-A / P1：按"块执行前"采样校准数据，并与旧的"执行后采样"做 A/B 对比。

背景（这是本脚本存在的唯一理由）：
    capture_calib.py 在 engine.step() 之后才读每层的输入缓冲，而引擎把状态
    原地更新（state 输出写回同一个缓冲），于是读到的 att_shift / ffn_shift / wkv
    是"更新后的状态"，不是该层这一步真正看到的输入；只有 x 与 v_first 恰好正确。
    用错误时序的数据算激活量化刻度，是不可信的。

本脚本：
    1) 复刻 Engine.step 的流程，但在**每层 execute() 之前**读取该层全部输入 → 真实输入；
    2) 同一 token 全部层执行完后再读一遍（= 旧脚本时序），与真实输入逐张量比对；
    3) 只落盘正确时序的校准数据，同时输出新旧差异统计（A/B 证据）。

输出：
    calib_pre/layerNN.npz   正确时序的输入（x / v_first / att_shift_0 / ffn_shift_0 / wkv_0）
    p0_sampling_ab.json     两种时序的差异统计（每张量的相对差异、以及状态"年龄"偏差）
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

MODEL_DIR = "/home/disk/models/rwkv7-2.9b"

# 小试文本：与最终 test 集分离，覆盖中文问答/叙述/少量代码，先用独立 16 段起步
PILOT_TEXT = (
    "System: 你是一名大学课程助手。\n\n"
    "User: 什么是梯度下降？\n\nAssistant: 梯度下降是一种迭代优化算法，沿着梯度反方向更新参数以最小化损失函数。\n\n"
    "User: 解释一下牛顿第一定律。\n\nAssistant: 任何物体都会保持静止或匀速直线运动状态，除非受到外力作用。\n\n"
    "User: 写一个判断素数的函数。\n\nAssistant: def is_prime(n):\n    if n < 2:\n        return False\n"
    "    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n\n"
    "User: 什么是熵？\n\nAssistant: 熵用于描述系统的无序程度；热力学第二定律指出孤立系统的熵不会自发减少。\n\n"
    "User: 光合作用的过程是什么？\n\nAssistant: 植物利用光能，把二氧化碳和水合成有机物并释放氧气。\n"
)


def read_buffer(acl, ptr, size):
    buf = np.empty(size // 4, np.float32)
    acl.rt.memcpy(buf.ctypes.data, size, ptr, size, 2)      # ACL_D2H
    return buf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=16, help="采样步数（状态年龄上限）")
    parser.add_argument("--out", default=os.path.join(MODEL_DIR, "calib_pre"))
    args = parser.parse_args()

    import acl
    from tokenizers import Tokenizer
    from rwkv7_serve2 import (Engine, LAYERS, X_BYTES, LOGITS_BYTES,
                              ACL_H2D, ACL_D2H)

    os.makedirs(args.out, exist_ok=True)
    tokenizer = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
    ids = tokenizer.encode(PILOT_TEXT).ids[: args.steps]
    print("小试文本 %d token，采样前 %d 步" % (len(tokenizer.encode(PILOT_TEXT).ids), len(ids)))

    engine = Engine(MODEL_DIR)                               # fp16 引擎
    store = [{n: [] for n in engine.layers[i].in_names} for i in range(LAYERS)]

    # 逐张量累计新旧时序的差异（不保存 post 值，避免内存翻倍）
    diff = {i: {n: {"abs": 0.0, "cnt": 0, "ref_abs": 0.0}
                for n in engine.layers[i].in_names} for i in range(LAYERS)}

    t0 = time.time()
    for step, tok in enumerate(ids):
        np.copyto(engine.x_host, engine.embedding[int(tok)])
        acl.rt.memcpy(engine.x_in_dev, X_BYTES, engine.x_host.ctypes.data,
                      X_BYTES, ACL_H2D)
        pre_this_step = []
        for i, om in enumerate(engine.layers):
            # (1) 执行前采样：这就是该层这一步真正的输入
            cur = {}
            for idx, name in enumerate(om.in_names):
                cur[name] = read_buffer(acl, om.in_dev[idx], om.in_sizes[idx])
                store[i][name].append(cur[name])
            pre_this_step.append(cur)
            om.execute()
        # (2) 全部执行完后再采一遍 = 旧脚本的时序，仅用于对比
        for i, om in enumerate(engine.layers):
            for idx, name in enumerate(om.in_names):
                post = read_buffer(acl, om.in_dev[idx], om.in_sizes[idx])
                pre = pre_this_step[i][name]
                d = diff[i][name]
                d["abs"] += float(np.abs(post - pre).sum())
                d["ref_abs"] += float(np.abs(pre).sum())
                d["cnt"] += pre.size
        engine.head.execute()
        acl.rt.memcpy(engine.logits.ctypes.data, LOGITS_BYTES,
                      engine.logits_dev, LOGITS_BYTES, ACL_D2H)
    print("采样 %d 步完成，用时 %.1f s" % (len(ids), time.time() - t0))

    # 落盘（只落正确时序）
    for i, data in enumerate(store):
        np.savez_compressed(os.path.join(args.out, "layer%02d.npz" % i),
                            **{k: np.stack(v) for k, v in data.items()})

    # A/B 差异报告
    report = {"model_dir": MODEL_DIR, "steps": len(ids),
              "pilot_text_sha1": hashlib.sha1(PILOT_TEXT.encode()).hexdigest(),
              "layers": {}}
    print("\n张量        前/后时序平均绝对差   相对量级")
    for i in range(LAYERS):
        rep = {}
        for name in store[i]:
            d = diff[i][name]
            mae = d["abs"] / max(d["cnt"], 1)
            ref = d["ref_abs"] / max(d["cnt"], 1)
            rel = mae / (ref + 1e-12)
            rep[name] = {"mae": round(mae, 6), "ref_mean_abs": round(ref, 6),
                         "relative": round(rel, 4)}
        report["layers"]["layer%02d" % i] = rep
        if i < 3 or i == LAYERS - 1:
            for name, v in rep.items():
                print("layer%-3d %-16s %-20.6f %.4f"
                      % (i, name, v["mae"], v["relative"]))
    with open(os.path.join(MODEL_DIR, "p0_sampling_ab.json"), "w") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print("\n已写出 %s 与 p0_sampling_ab.json" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
