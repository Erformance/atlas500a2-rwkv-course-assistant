#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对比两个 .om 的单次 execute 耗时（同一输入反复跑）。

用法: python rwkv7_cmp_om.py a.om b.om [重复次数]
"""

import sys
import time

import numpy as np

import acl


def check(ret):
    if isinstance(ret, tuple):
        ret = ret[-1]
    if ret != 0:
        raise RuntimeError("acl 调用失败: %s" % (ret,))


class Om:
    def __init__(self, path):
        self.path = path
        self.mid, ret = acl.mdl.load_from_file(path)
        check(ret)
        self.desc = acl.mdl.create_desc()
        check(acl.mdl.get_desc(self.desc, self.mid))
        n_in = acl.mdl.get_num_inputs(self.desc)
        n_out = acl.mdl.get_num_outputs(self.desc)
        self.in_sizes = [acl.mdl.get_input_size_by_index(self.desc, i) for i in range(n_in)]
        self.out_sizes = [acl.mdl.get_output_size_by_index(self.desc, i) for i in range(n_out)]
        self.in_dev = [acl.rt.malloc(s, 0)[0] for s in self.in_sizes]
        self.out_dev = [acl.rt.malloc(s, 0)[0] for s in self.out_sizes]
        self.in_ds = acl.mdl.create_dataset()
        for ptr, size in zip(self.in_dev, self.in_sizes):
            self.in_ds, _ = acl.mdl.add_dataset_buffer(self.in_ds, acl.create_data_buffer(ptr, size))
        self.out_ds = acl.mdl.create_dataset()
        for ptr, size in zip(self.out_dev, self.out_sizes):
            self.out_ds, _ = acl.mdl.add_dataset_buffer(self.out_ds, acl.create_data_buffer(ptr, size))
        self.bytes_in = sum(self.in_sizes)

    def run(self):
        check(acl.mdl.execute(self.mid, self.in_ds, self.out_ds))


def bench(om, repeat):
    om.run()
    t0 = time.time()
    for _ in range(repeat):
        om.run()
    return (time.time() - t0) / repeat * 1000.0


def main():
    paths = sys.argv[1:3]
    repeat = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    check(acl.init())
    check(acl.rt.set_device(0))
    ctx, ret = acl.rt.create_context(0)
    check(ret)
    for path in paths:
        om = Om(path)
        ms = bench(om, repeat)
        print("%-34s execute %7.3f ms   (输入字节 %d)" % (path.split("/")[-1], ms, om.bytes_in))
    acl.rt.destroy_context(ctx)
    acl.rt.reset_device(0)
    acl.finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
