#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对比同一层的 fp16 版 .om 与 int8 量化版 .om 的数值差异。

用法: python cmp_layer_numeric.py a.om b.om
"""

import sys

import numpy as np

import acl


def check(ret):
    if isinstance(ret, tuple):
        ret = ret[-1]
    if ret != 0:
        raise RuntimeError("acl 失败: %s" % (ret,))


class Om:
    def __init__(self, path):
        self.path = path
        self.mid, ret = acl.mdl.load_from_file(path)
        check(ret)
        self.desc = acl.mdl.create_desc()
        check(acl.mdl.get_desc(self.desc, self.mid))
        n_in = acl.mdl.get_num_inputs(self.desc)
        n_out = acl.mdl.get_num_outputs(self.desc)
        self.in_names = [acl.mdl.get_input_name_by_index(self.desc, i) for i in range(n_in)]
        self.out_names = [acl.mdl.get_output_name_by_index(self.desc, i) for i in range(n_out)]
        self.in_sizes = [acl.mdl.get_input_size_by_index(self.desc, i) for i in range(n_in)]
        self.out_sizes = [acl.mdl.get_output_size_by_index(self.desc, i) for i in range(n_out)]
        self.dev_in = [acl.rt.malloc(s, 0)[0] for s in self.in_sizes]
        self.dev_out = [acl.rt.malloc(s, 0)[0] for s in self.out_sizes]
        self.in_ds = acl.mdl.create_dataset()
        for p, s in zip(self.dev_in, self.in_sizes):
            self.in_ds, _ = acl.mdl.add_dataset_buffer(self.in_ds, acl.create_data_buffer(p, s))
        self.out_ds = acl.mdl.create_dataset()
        for p, s in zip(self.dev_out, self.out_sizes):
            self.out_ds, _ = acl.mdl.add_dataset_buffer(self.out_ds, acl.create_data_buffer(p, s))

    def run(self, feeds):
        for i, name in enumerate(self.in_names):
            arr = np.ascontiguousarray(feeds[i], dtype=np.float32)
            check(acl.rt.memcpy(self.dev_in[i], self.in_sizes[i], arr.ctypes.data,
                                arr.nbytes, 1))
        check(acl.mdl.execute(self.mid, self.in_ds, self.out_ds))
        outs = []
        for i in range(len(self.out_sizes)):
            buf = np.empty(self.out_sizes[i] // 4, np.float32)
            check(acl.rt.memcpy(buf.ctypes.data, self.out_sizes[i], self.dev_out[i],
                                self.out_sizes[i], 2))
            outs.append(buf)
        return outs


def main():
    check(acl.init())
    check(acl.rt.set_device(0))
    ctx, ret = acl.rt.create_context(0)
    check(ret)
    a_om, b_om = Om(sys.argv[1]), Om(sys.argv[2])

    rng = np.random.default_rng(0)
    feeds = []
    for size in a_om.in_sizes:
        n = size // 4
        if n == 2560:                    # shift state：小值
            feeds.append(rng.normal(0, 0.5, n).astype(np.float32))
        elif n == 40 * 64 * 64:          # wkv state
            feeds.append(rng.normal(0, 0.1, n).astype(np.float32))
        else:                            # x / v_first
            feeds.append(rng.normal(0, 1.0, n).astype(np.float32))
    feeds = [f.reshape(-1) for f in feeds]

    out_a = a_om.run(feeds)
    out_b = b_om.run(feeds)

    labels = list(a_om.out_names)
    print("%-46s %-12s %-12s %-10s" % ("输出张量", "fp16 幅度", "int8 幅度", "相对误差"))
    for name, x, y in zip(labels, out_a, out_b):
        scale = float(np.abs(x).mean()) + 1e-8
        rel = float(np.abs(x - y).mean() / scale)
        print("%-46s %-12.4f %-12.4f %-10.4f" % (name[:44], float(np.abs(x).mean()),
                                                 float(np.abs(y).mean()), rel))
    acl.rt.destroy_context(ctx)
    acl.rt.reset_device(0)
    acl.finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
