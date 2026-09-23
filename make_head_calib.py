#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成输出头（head）的校准数据。

head 的输入就是第 31 层的 x 输出，所以直接用 calib/layer31.npz 里的真实输入
跑一遍 fp16 的 layer31.om，把 x 输出存成 calib/head.npz 即可。

用法: python make_head_calib.py [--calib-dir calib_pre] [--out-npz calib_pre/head.npz]
"""

import argparse
import os
import re
import sys

import numpy as np

MODEL_DIR = "/home/disk/models/rwkv7-2.9b"
sys.path.insert(0, MODEL_DIR)

import acl                                     # noqa: E402
from cmp_layer_numeric import Om, check        # noqa: E402


def norm(name):
    return re.sub(r"\.\d+$", "", name.split(":")[-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calib-dir", default=os.path.join(MODEL_DIR, "calib"))
    parser.add_argument("--out-npz", default=None)
    parser.add_argument("--layer", type=int, default=31)
    args = parser.parse_args()
    out_npz = args.out_npz or os.path.join(args.calib_dir, "head.npz")

    check(acl.init())
    check(acl.rt.set_device(0))
    ctx, ret = acl.rt.create_context(0)
    check(ret)

    om = Om(os.path.join(MODEL_DIR, "layer%d.om" % args.layer))
    data = np.load(os.path.join(args.calib_dir, "layer%02d.npz" % args.layer))
    steps = data["x"].shape[0]

    outs = []
    for t in range(steps):
        feeds = [np.ascontiguousarray(data[norm(n)][t].reshape(-1), np.float32)
                 for n in om.in_names]
        outs.append(om.run(feeds)[0])          # 第一个输出就是 x
    stacked = np.stack(outs)
    print("head 校准样本 %d 组，形状 %s，峰值 %.3f"
          % (stacked.shape[0], stacked.shape[1:], float(np.abs(stacked).max())))
    np.savez_compressed(out_npz, x=stacked)
    print("已写出 %s" % out_npz)

    acl.rt.destroy_context(ctx)
    acl.rt.reset_device(0)
    acl.finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
