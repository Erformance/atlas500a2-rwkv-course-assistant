#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用真实抓取的输入对比两个 .om（fp16 vs int8）的输出误差。

用法: python cmp_layer_real.py a.om b.om calib/layer07.npz [步号]
"""

import os
import re
import sys

import numpy as np

sys.path.insert(0, "/home/disk/models/rwkv7-2.9b")

import acl                                     # noqa: E402
from cmp_layer_numeric import Om, check        # noqa: E402


def norm(name):
    return re.sub(r"\.\d+$", "", name.split(":")[-1])


def main():
    a_path, b_path, npz_path = sys.argv[1], sys.argv[2], sys.argv[3]
    step = int(sys.argv[4]) if len(sys.argv) > 4 else 0

    check(acl.init())
    check(acl.rt.set_device(0))
    ctx, ret = acl.rt.create_context(0)
    check(ret)

    a_om, b_om = Om(a_path), Om(b_path)
    if a_om.in_names != b_om.in_names:
        print("注意：两个模型的输入名不一致")

    data = np.load(npz_path)
    feeds = []
    for name in a_om.in_names:
        key = norm(name)
        arr = data[key][step]
        feeds.append(np.ascontiguousarray(arr.reshape(-1), dtype=np.float32))

    out_a = a_om.run(feeds)
    out_b = b_om.run(feeds)

    print("%-34s %-11s %-11s %-9s" % ("输出张量", "fp16 幅度", "int8 幅度", "相对误差"))
    worst = 0.0
    for name, x, y in zip(a_om.out_names, out_a, out_b):
        scale = float(np.abs(x).mean()) + 1e-8
        rel = float(np.abs(x - y).mean() / scale)
        worst = max(worst, rel)
        print("%-34s %-11.4f %-11.4f %-9.4f" % (norm(name)[:32], float(np.abs(x).mean()),
                                                float(np.abs(y).mean()), rel))
    print("最大相对误差 %.4f" % worst)
    acl.rt.destroy_context(ctx)
    acl.rt.reset_device(0)
    acl.finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
