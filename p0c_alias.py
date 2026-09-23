#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P0-C：原地（状态输入输出别名）与双缓冲（ping-pong）执行的等价性测试。

引擎把状态输出写回输入缓冲（原地更新）。本测试固定同一权重、同一 x/v_first 输入、
同一初始状态，只改变状态缓冲的用法，连续执行 N 步后比较最终状态与逐步快照：
逐位一致 ⇒ 原地执行在本实现上是安全契约，引擎的零拷贝设计成立。

用法: python p0c_alias.py --layer 7 --steps 1,8,128
输出: p0c_alias.json
"""

import argparse
import json
import os
import sys

import numpy as np

MODEL_DIR = "/home/disk/models/rwkv7-2.9b"
sys.path.insert(0, MODEL_DIR)

import acl                                              # noqa: E402
from rwkv7_serve2 import OmModel, ACL_H2D, ACL_D2H       # noqa: E402


def d2h(ptr, size):
    buf = np.empty(size // 4, np.float32)
    acl.rt.memcpy(buf.ctypes.data, size, ptr, size, ACL_D2H)
    return buf


def base_name(name):
    """att_shift_out_0 -> att_shift_0；x -> x"""
    return name.replace("_out_", "_").replace("_out", "")


def run_variant(om, mode, steps, seed=0):
    """mode='inplace'：状态输出写回状态输入缓冲；mode='pingpong'：写到独立缓冲再搬回。"""
    rng = np.random.default_rng(seed)
    in_by_name, in_ptrs = {}, []
    for idx, name in enumerate(om.in_names):
        size = om.in_sizes[idx]
        ptr = acl.rt.malloc(size, 0)[0]
        if name.startswith(("x", "v_first")):
            arr = rng.normal(0, 1.0, size // 4).astype(np.float32)
        else:
            arr = np.zeros(size // 4, np.float32)
        acl.rt.memcpy(ptr, size, arr.ctypes.data, size, ACL_H2D)
        in_by_name[name] = ptr
        in_ptrs.append(ptr)
    scratch = {name: acl.rt.malloc(s, 0)[0] for name, s in zip(om.out_names, om.out_sizes)}

    def attach(out_ptrs):
        om.in_ds = acl.mdl.create_dataset()
        for p, s in zip(in_ptrs, om.in_sizes):
            om.in_ds, _ = acl.mdl.add_dataset_buffer(
                om.in_ds, acl.create_data_buffer(p, s))
        om.out_ds = acl.mdl.create_dataset()
        for p, s in zip(out_ptrs, om.out_sizes):
            om.out_ds, _ = acl.mdl.add_dataset_buffer(
                om.out_ds, acl.create_data_buffer(p, s))

    state_in = [n for n in om.in_names if not n.startswith(("x", "v_first"))]
    snaps = []
    for _ in range(steps):
        out_ptrs = []
        for oname, size in zip(om.out_names, om.out_sizes):
            base = base_name(oname)
            if mode == "inplace" and base in in_by_name:
                out_ptrs.append(in_by_name[base])          # 原地：写回输入缓冲
            else:
                out_ptrs.append(scratch[oname])
        attach(out_ptrs)
        om.execute()
        if mode == "pingpong":
            for oname, size in zip(om.out_names, om.out_sizes):
                base = base_name(oname)
                if base in in_by_name:                     # 显式 D2D 搬回，作为下一步输入
                    acl.rt.memcpy(in_by_name[base], size, scratch[oname], size, 1)
        snaps.append([d2h(in_by_name[n], om.in_sizes[om.in_names.index(n)])
                      for n in state_in])
    return snaps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=7)
    parser.add_argument("--steps", default="1,8,128")
    parser.add_argument("--out", default=os.path.join(MODEL_DIR, "p0c_alias.json"))
    args = parser.parse_args()
    step_list = [int(x) for x in args.steps.split(",")]

    # 注意：本脚本直接用 OmModel，不像 Engine 那样自带初始化，必须自己建 context，
    # 否则 acl.mdl.load_from_file 会失败（145001）。
    acl.init()
    acl.rt.set_device(0)
    ctx, ret = acl.rt.create_context(0)
    if ret != 0:
        raise RuntimeError("create_context 失败: %s" % (ret,))

    report = {"layer": args.layer, "runs": []}
    for steps in step_list:
        a = OmModel(acl, os.path.join(MODEL_DIR, "layer%d.om" % args.layer))
        snaps_a = run_variant(a, "inplace", steps)
        b = OmModel(acl, os.path.join(MODEL_DIR, "layer%d.om" % args.layer))
        snaps_b = run_variant(b, "pingpong", steps)
        final_a, final_b = snaps_a[-1], snaps_b[-1]
        diffs = [float(np.abs(x - y).max()) for x, y in zip(final_a, final_b)]
        equal = all(np.array_equal(x, y) for x, y in zip(final_a, final_b))
        step_equal = all(np.array_equal(x, y)
                         for sa, sb in zip(snaps_a, snaps_b)
                         for x, y in zip(sa, sb))
        report["runs"].append({"steps": steps, "max_abs_diff": diffs,
                               "final_bitwise_equal": equal,
                               "all_steps_bitwise_equal": step_equal})
        print("步数 %-4d 最终状态最大差 %s ｜ 逐位一致 %s ｜ 逐步一致 %s"
              % (steps, " ".join("%.3e" % d for d in diffs), equal, step_equal))
    report["conclusion"] = (
        "两种执行方式逐位一致：原地更新是安全契约，零拷贝设计成立"
        if all(r["all_steps_bitwise_equal"] for r in report["runs"])
        else "存在差异：引擎的输入输出别名假设需要修正")
    with open(args.out, "w") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print("\n%s" % report["conclusion"])
    acl.rt.destroy_context(ctx)
    acl.rt.reset_device(0)
    acl.finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
