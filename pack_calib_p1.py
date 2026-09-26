#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 capture_calib_p1.py 的 memmap 打包成 quantize_layer_calib.py 能直接吃的 npz。

逐层打包：每层只把一张量族读进内存（几 MB ~ 几十 MB），避免 96 组样本 × 32 层
（约 2.2GB）一次性驻留。

用法: python pack_calib_p1.py --raw calib_p1_raw --out calib_p1
输出: calib_p1/layerNN.npz（键名与 capture_calib_pre.py 一致，量化脚本无需改动）
"""

import argparse
import glob
import os

import numpy as np

MODEL_DIR = os.environ.get("RWKV_MODEL_DIR", "/home/disk/models/rwkv7-2.9b")
LAYERS = 32


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default=os.path.join(MODEL_DIR, "calib_p1_raw"))
    parser.add_argument("--out", default=os.path.join(MODEL_DIR, "calib_p1"))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    total_bytes = 0
    for i in range(LAYERS):
        prefix = "layer%02d_" % i
        files = sorted(glob.glob(os.path.join(args.raw, prefix + "*.npy")))
        if not files:
            raise SystemExit("缺少 %s 的采样文件（先跑 capture_calib_p1.py）" % prefix)
        data, steps = {}, None
        for path in files:
            name = os.path.basename(path)[len(prefix):-len(".npy")]
            arr = np.load(path, mmap_mode="r")          # 只映射，读的时候才进内存
            if steps is None:
                steps = arr.shape[0]
            elif arr.shape[0] != steps:
                raise SystemExit("%s 样本数 %d 与其它张量 %d 不一致"
                                 % (name, arr.shape[0], steps))
            data[name] = np.asarray(arr, dtype=np.float32)
        out = os.path.join(args.out, "layer%02d.npz" % i)
        np.savez(out, **data)                            # 不压缩：量化脚本按需取键
        size = os.path.getsize(out)
        total_bytes += size
        print("layer%02d：%2d 组 × %-40s → %6.1f MB"
              % (i, steps, ",".join(sorted(data)), size / 1e6))
    print("打包完成：32 层，合计 %.2f GB" % (total_bytes / 1e9))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
