#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成输出头（head）的校准数据。

head 的输入就是第 31 层的 x 输出，所以直接用 calib/layer31.npz 里的真实输入
跑一遍 fp16 的 layer31.om，把 x 输出存成 calib/head.npz 即可。

用法: python make_head_calib.py
"""

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
    check(acl.init())
    check(acl.rt.set_device(0))
    ctx, ret = acl.rt.create_context(0)
    check(ret)

    om = Om(os.path.join(MODEL_DIR, "layer31.om"))
    data = np.load(os.path.join(MODEL_DIR, "calib", "layer31.npz"))
    steps = data["x"].shape[0]

    outs = []
    for t in range(steps):
        feeds = [np.ascontiguousarray(data[norm(n)][t].reshape(-1), np.float32)
                 for n in om.in_names]
        outs.append(om.run(feeds)[0])          # 第一个输出就是 x
    stacked = np.stack(outs)
    print("head 校准样本 %d 组，形状 %s，峰值 %.3f"
          % (stacked.shape[0], stacked.shape[1:], float(np.abs(stacked).max())))
    np.savez_compressed(os.path.join(MODEL_DIR, "calib", "head.npz"), x=stacked)
    print("已写出 calib/head.npz")

    acl.rt.destroy_context(ctx)
    acl.rt.reset_device(0)
    acl.finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
