#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RWKV-7 2.9B NPU 推理耗时归因测量。

分别测量：
  1) 单层 .om 的纯 execute 时间
  2) 单层 .om 含 host<->device 搬运的总时间
  3) head .om 的 execute 时间
并据此推算单 token 的理论构成，与实测 0.21s/token 对照。
"""

import os
import time

import numpy as np

import acl

MODEL_DIR = "/home/disk/models/rwkv7-2.9b"
HIDDEN, HEADS, HEAD_DIM = 2560, 40, 64
REPEAT = 30


def check(ret, what):
    if isinstance(ret, tuple):
        ret = ret[-1]
    if ret != 0:
        raise RuntimeError("%s failed: %s" % (what, ret))


class Om:
    def __init__(self, path):
        self.mid, ret = acl.mdl.load_from_file(path)
        check(ret, "load " + path)
        self.desc = acl.mdl.create_desc()
        check(acl.mdl.get_desc(self.desc, self.mid), "get_desc")
        n_in = acl.mdl.get_num_inputs(self.desc)
        n_out = acl.mdl.get_num_outputs(self.desc)
        self.in_names = [acl.mdl.get_input_name_by_index(self.desc, i) for i in range(n_in)]
        self.out_names = [acl.mdl.get_output_name_by_index(self.desc, i) for i in range(n_out)]
        self.in_sizes = [acl.mdl.get_input_size_by_index(self.desc, i) for i in range(n_in)]
        self.out_sizes = [acl.mdl.get_output_size_by_index(self.desc, i) for i in range(n_out)]
        self.dev_in = [acl.rt.malloc(s, 0)[0] for s in self.in_sizes]
        self.dev_out = [acl.rt.malloc(s, 0)[0] for s in self.out_sizes]
        self.in_ds = acl.mdl.create_dataset()
        for ptr, size in zip(self.dev_in, self.in_sizes):
            self.in_ds, ret = acl.mdl.add_dataset_buffer(
                self.in_ds, acl.create_data_buffer(ptr, size))
        self.out_ds = acl.mdl.create_dataset()
        for ptr, size in zip(self.dev_out, self.out_sizes):
            self.out_ds, ret = acl.mdl.add_dataset_buffer(
                self.out_ds, acl.create_data_buffer(ptr, size))
        self.feeds = [np.zeros(s // 4, np.float32) for s in self.in_sizes]
        self.outs = [np.empty(s // 4, np.float32) for s in self.out_sizes]
        self.bytes_in = sum(self.in_sizes)
        self.bytes_out = sum(self.out_sizes)

    def exec_only(self):
        check(acl.mdl.execute(self.mid, self.in_ds, self.out_ds), "execute")

    def exec_with_copy(self):
        for i, size in enumerate(self.in_sizes):
            check(acl.rt.memcpy(self.dev_in[i], size, self.feeds[i].ctypes.data,
                                size, 1), "h2d")
        check(acl.mdl.execute(self.mid, self.in_ds, self.out_ds), "execute")
        for i, size in enumerate(self.out_sizes):
            check(acl.rt.memcpy(self.outs[i].ctypes.data, size, self.dev_out[i],
                                size, 2), "d2h")


def bench(fn, repeat=REPEAT):
    fn()  # warmup
    t0 = time.time()
    for _ in range(repeat):
        fn()
    return (time.time() - t0) / repeat * 1000.0


def main():
    check(acl.init(), "acl.init")
    check(acl.rt.set_device(0), "set_device")
    ctx, ret = acl.rt.create_context(0)
    check(ret, "create_context")

    layer = Om(os.path.join(MODEL_DIR, "layer1.om"))
    head = Om(os.path.join(MODEL_DIR, "head.om"))

    print("layer1: 输入 %d 个 %d 字节，输出 %d 个 %d 字节"
          % (len(layer.in_sizes), layer.bytes_in, len(layer.out_sizes), layer.bytes_out))
    print("head  : 输入 %d 字节，输出 %d 字节" % (head.bytes_in, head.bytes_out))
    print("-" * 62)

    t_layer_exec = bench(layer.exec_only)
    t_layer_full = bench(layer.exec_with_copy)
    t_head_exec = bench(head.exec_only)
    t_head_full = bench(head.exec_with_copy)

    print("单层 execute         : %7.3f ms" % t_layer_exec)
    print("单层 execute+搬运     : %7.3f ms  (搬运 %.3f ms)"
          % (t_layer_full, t_layer_full - t_layer_exec))
    print("head execute         : %7.3f ms" % t_head_exec)
    print("head execute+搬运     : %7.3f ms" % t_head_full)
    print("-" * 62)

    per_token = 32 * t_layer_full + t_head_full
    print("按 32 层 + head 推算单 token: %.1f ms → %.2f token/s"
          % (per_token, 1000.0 / per_token))
    print("实测对照: 210 ms/token → 4.77 token/s")
    print("-" * 62)
    print("权重字节: 单层 %d B（%.1f MB），32 层 %.2f GB"
          % (layer.bytes_in, layer.bytes_in / 1e6,
             (layer.bytes_in * 1) / 1e9))
    bw_floor = 5.54e9 / 51.2e9 * 1000
    print("纯带宽下限（按 5.54GB 权重 / 51.2GB/s）: %.1f ms/token → %.2f token/s"
          % (bw_floor, 1000.0 / bw_floor))

    acl.rt.destroy_context(ctx)
    acl.rt.reset_device(0)
    acl.finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
