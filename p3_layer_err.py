#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P3 局部度量（单层）：同一份真实校准输入下，int8 层与 fp16 层的输出误差。

协议 §5 要比较"同输入单步输出 MSE""归一化状态 Frobenius 误差"这类**局部指标**
与"完整短 rollout KL"的预测力。本脚本用 calib_p1 里真实抓到的输入，在几个采样位置上
跑同一层的 fp16 与 int8 版本，按输出张量分别给相对误差（含 wkv 状态张量）。

用法: python p3_layer_err.py --layer 9 --suffix _p1_q --calib calib_p1 --out /tmp/x.json
输出: JSON（x/att_shift/ffn_shift/wkv 各自相对误差 + 最大值）
"""

import argparse
import json
import os
import re
import sys

import numpy as np

MODEL_DIR = "/home/disk/models/rwkv7-2.9b"
sys.path.insert(0, MODEL_DIR)


def norm(name):
    return re.sub(r"\.\d+$", "", name.split(":")[-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--suffix", default="_p1_q")
    parser.add_argument("--calib", default="calib_p1")
    parser.add_argument("--steps", default="0,1,2", help="用哪几个校准样本取平均")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import acl
    from cmp_layer_numeric import Om, check

    check(acl.init())
    check(acl.rt.set_device(0))
    ctx, ret = acl.rt.create_context(0)
    check(ret)

    om_fp16 = Om(os.path.join(MODEL_DIR, "layer%d.om" % args.layer))
    om_int8 = Om(os.path.join(MODEL_DIR, "layer%d%s.om" % (args.layer, args.suffix)))
    data = np.load(os.path.join(MODEL_DIR, args.calib, "layer%02d.npz" % args.layer))
    steps = [int(x) for x in args.steps.split(",") if x.strip()]

    per_out = {}
    for step in steps:
        feeds = [np.ascontiguousarray(data[norm(n)][step].reshape(-1), np.float32)
                 for n in om_fp16.in_names]
        out_a = om_fp16.run(feeds)
        out_b = om_int8.run(feeds)
        for name, u, v in zip(om_fp16.out_names, out_a, out_b):
            key = norm(name)
            scale = float(np.abs(u).mean()) + 1e-8
            per_out.setdefault(key, []).append(float(np.abs(u - v).mean() / scale))

    rel = {k: round(float(np.mean(v)), 6) for k, v in per_out.items()}
    result = {"layer": args.layer, "suffix": args.suffix,
              "steps": steps, "rel_err": rel,
              "rel_err_max": round(max(rel.values()), 6) if rel else None}
    with open(args.out, "w") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False))

    acl.rt.destroy_context(ctx)
    acl.rt.reset_device(0)
    acl.finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
