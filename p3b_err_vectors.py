#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P3 方向指标第一步：在 NPU 上实测每层的**误差向量** δᵢ（不是只测相对误差）。

同一份真实校准输入下，跑 fp16 层与 int8 层（`_p1_q`），把逐元素差存下来，
供 `p3b_dir_metric.py` 计算"输出加权方向灵敏度" γᵢ = |gᵢ · δᵢ|。

每层独立子进程（同进程反复建 Om 会累积设备内存）。

用法:
  python p3b_err_vectors.py                      # 32 层全跑
  python p3b_err_vectors.py --layer 9 --out /tmp/x.npz
输出: p3b_err_vec/layerNN.npz，键为 "<张量名>_<步号>"（float32）
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

MODEL_DIR = os.environ.get("RWKV_MODEL_DIR", "/home/disk/models/rwkv7-2.9b")
sys.path.insert(0, MODEL_DIR)
PY = "/home/disk/miniconda3/envs/npu22/bin/python"


def norm(name):
    return re.sub(r"\.\d+$", "", name.split(":")[-1])


def one_layer(layer, suffix, calib, steps, out):
    import acl
    from cmp_layer_numeric import Om, check

    check(acl.init())
    check(acl.rt.set_device(0))
    ctx, ret = acl.rt.create_context(0)
    check(ret)
    om_fp16 = Om(os.path.join(MODEL_DIR, "layer%d.om" % layer))
    om_int8 = Om(os.path.join(MODEL_DIR, "layer%d%s.om" % (layer, suffix)))
    data = np.load(os.path.join(MODEL_DIR, calib, "layer%02d.npz" % layer))
    payload = {}
    for step in steps:
        feeds = [np.ascontiguousarray(data[norm(n)][step].reshape(-1), np.float32)
                 for n in om_fp16.in_names]
        a = om_fp16.run(feeds)
        b = om_int8.run(feeds)
        for name, u, v in zip(om_fp16.out_names, a, b):
            payload["%s_%d" % (norm(name), step)] = (u - v).astype(np.float32)
    np.savez(out, **payload)
    acl.rt.destroy_context(ctx)
    acl.rt.reset_device(0)
    acl.finalize()
    return {k: float(np.abs(v).max()) for k, v in payload.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", default="_p1_q")
    parser.add_argument("--calib", default="calib_p1")
    parser.add_argument("--steps", default="0,1,2")
    parser.add_argument("--layer", type=int, default=None, help="只跑这一层（子进程模式）")
    parser.add_argument("--out", default=None)
    parser.add_argument("--out-dir", default=os.path.join(MODEL_DIR, "p3b_err_vec"))
    args = parser.parse_args()
    steps = [int(x) for x in args.steps.split(",") if x.strip()]

    if args.layer is not None:
        mx = one_layer(args.layer, args.suffix, args.calib, steps, args.out)
        print(json.dumps({"layer": args.layer, "absmax": mx}, ensure_ascii=False))
        return 0

    os.makedirs(args.out_dir, exist_ok=True)
    ok = 0
    for i in range(32):
        out = os.path.join(args.out_dir, "layer%02d.npz" % i)
        if os.path.exists(out):
            print("layer%-2d 跳过（已有）" % i, flush=True)
            ok += 1
            continue
        t0 = time.time()
        r = subprocess.run([PY, os.path.abspath(__file__), "--layer", str(i),
                            "--suffix", args.suffix, "--calib", args.calib,
                            "--steps", args.steps, "--out", out],
                           capture_output=True, text=True)
        if os.path.exists(out):
            print("layer%-2d %.1fs %s" % (i, time.time() - t0, (r.stdout or "").strip()[:120]),
                  flush=True)
            ok += 1
        else:
            print("layer%-2d 失败：%s" % (i, (r.stderr or r.stdout)[-160:]), flush=True)
    print("\n完成 %d/32 层，输出目录 %s" % (ok, args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
