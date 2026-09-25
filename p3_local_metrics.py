#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P3 局部度量驱动：32 层逐个跑 p3_layer_err.py（每层独立子进程），汇总成 p3_local.json。

每层用同一份真实校准输入（calib_p1 的 3 个采样位置）比 fp16 与 int8 的输出，
得到"单步输出误差"与"状态张量误差"两类局部指标，供与 rollout KL 做相关性比较。

用法: python p3_local_metrics.py [--suffix _p1_q] [--calib calib_p1] [--out p3_local.json]
"""

import argparse
import json
import os
import subprocess
import sys
import time

MODEL_DIR = "/home/disk/models/rwkv7-2.9b"
PY = "/home/disk/miniconda3/envs/npu22/bin/python"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", default="_p1_q")
    parser.add_argument("--calib", default="calib_p1")
    parser.add_argument("--out", default=os.path.join(MODEL_DIR, "p3_local.json"))
    parser.add_argument("--layers", default=",".join(str(i) for i in range(32)))
    args = parser.parse_args()

    parts = os.path.join(MODEL_DIR, "p3_local_parts")
    os.makedirs(parts, exist_ok=True)
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    results = {}
    if os.path.exists(args.out):
        with open(args.out) as fh:
            results = json.load(fh).get("layers", {})

    for i in layers:
        part = os.path.join(parts, "layer%02d.json" % i)
        if not os.path.exists(part):
            cmd = [PY, os.path.join(MODEL_DIR, "p3_layer_err.py"),
                   "--layer", str(i), "--suffix", args.suffix,
                   "--calib", args.calib, "--steps", "0,1,2", "--out", part]
            t0 = time.time()
            r = subprocess.run(cmd, capture_output=True, text=True)
            if not os.path.exists(part):
                print("layer%-2d 失败：%s" % (i, (r.stderr or r.stdout)[-200:]), flush=True)
                continue
            print("layer%-2d %.1fs %s" % (i, time.time() - t0,
                                          (r.stdout or "").strip()[:150]), flush=True)
        with open(part) as fh:
            d = json.load(fh)
        results["%d" % i] = d
        with open(args.out, "w") as fh:
            json.dump({"suffix": args.suffix, "calib": args.calib, "layers": results},
                      fh, ensure_ascii=False, indent=2)

    print("\n汇总：%d 层，已写出 %s" % (len(results), args.out))
    for name in sorted(results, key=lambda x: -results[x]["rel_err_max"])[:5]:
        print("  误差最大 %-3s %s" % (name, results[name]["rel_err"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
