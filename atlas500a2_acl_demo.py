#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Atlas 500 A2（昇腾 310B）pyACL 最小推理验证。

用途：绕过 torch_npu，直接用设备自带的官方推理链路（ATC 编译 .om + pyACL 执行），
验证 NPU 能否真正完成一次算子计算。

用法：
    source /home/disk/cann80base/ascend-toolkit/set_env.sh
    python atlas500a2_acl_demo.py /home/disk/tmp/tiny.om \
        --input /home/disk/tmp/tiny_in.npy --ref /home/disk/tmp/tiny_ref.npy
"""

import argparse
import sys
import time

import numpy as np

import acl

ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2


def retcode(value):
    """pyACL 部分接口返回 (obj, ret) 元组，统一取出返回码。"""
    if isinstance(value, tuple):
        return value[-1]
    return value


def check(ret, what):
    ret = retcode(ret)
    if ret != 0:
        raise RuntimeError("%s 失败: ret=%s msg=%s" % (what, ret, acl.get_recent_err_msg()))


def run_once(model_id, desc, dev_in, in_size, dev_out, out_size, host_in):
    check(
        acl.rt.memcpy(
            dev_in, in_size, host_in.ctypes.data, in_size, ACL_MEMCPY_HOST_TO_DEVICE
        ),
        "H2D memcpy",
    )

    in_ds = acl.mdl.create_dataset()
    check(
        acl.mdl.add_dataset_buffer(in_ds, acl.create_data_buffer(dev_in, in_size)),
        "加输入 dataset",
    )
    out_ds = acl.mdl.create_dataset()
    check(
        acl.mdl.add_dataset_buffer(out_ds, acl.create_data_buffer(dev_out, out_size)),
        "加输出 dataset",
    )

    host_out = np.zeros(out_size // 4, dtype=np.float32)
    t0 = time.time()
    check(acl.mdl.execute(model_id, in_ds, out_ds), "acl.mdl.execute")
    torch_like_sync = time.time()
    check(
        acl.rt.memcpy(
            host_out.ctypes.data, out_size, dev_out, out_size, ACL_MEMCPY_DEVICE_TO_HOST
        ),
        "D2H memcpy",
    )
    t1 = time.time()

    acl.mdl.destroy_dataset(in_ds)
    acl.mdl.destroy_dataset(out_ds)
    return host_out, (torch_like_sync - t0) * 1000.0, (t1 - t0) * 1000.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("om", help=".om 模型路径")
    parser.add_argument("--input", help="输入向量 .npy（默认用 arange 填充）")
    parser.add_argument("--ref", help="CPU 参考输出 .npy，用于数值比对")
    parser.add_argument("--repeat", type=int, default=1, help="重复执行次数")
    parser.add_argument("--device", type=int, default=0, help="设备号")
    args = parser.parse_args()

    check(acl.init(), "acl.init")
    print("[1] acl.init ok")
    check(acl.rt.set_device(args.device), "acl.rt.set_device")
    print("[2] set_device(%d) ok" % args.device)
    context, ret = acl.rt.create_context(args.device)
    check(ret, "acl.rt.create_context")
    print("[3] create_context ok")

    model_id, ret = acl.mdl.load_from_file(args.om)
    check(ret, "acl.mdl.load_from_file")
    print("[4] 模型加载 ok，model_id=%s" % model_id)

    desc = acl.mdl.create_desc()
    check(acl.mdl.get_desc(desc, model_id), "acl.mdl.get_desc")
    n_in = acl.mdl.get_num_inputs(desc)
    n_out = acl.mdl.get_num_outputs(desc)
    in_size = acl.mdl.get_input_size_by_index(desc, 0)
    out_size = acl.mdl.get_output_size_by_index(desc, 0)
    print("[5] 模型描述: 输入 %d 个 / 输出 %d 个, 输入 %d 字节, 输出 %d 字节"
          % (n_in, n_out, in_size, out_size))

    if args.input:
        host_in = np.load(args.input).astype(np.float32).ravel()
    else:
        host_in = np.arange(in_size // 4, dtype=np.float32)
    if host_in.nbytes != in_size:
        raise SystemExit("输入大小不匹配: npy=%d 字节, 模型需要 %d 字节" % (host_in.nbytes, in_size))

    dev_in, ret = acl.rt.malloc(in_size, 0)
    check(ret, "acl.rt.malloc(输入)")
    dev_out, ret = acl.rt.malloc(out_size, 0)
    check(ret, "acl.rt.malloc(输出)")

    try:
        last = None
        for i in range(1, args.repeat + 1):
            host_out, exec_ms, total_ms = run_once(
                model_id, desc, dev_in, in_size, dev_out, out_size, host_in
            )
            last = host_out
            print("[6] 第 %d 次执行: NPU 执行 %.3f ms, 含搬运 %.3f ms" % (i, exec_ms, total_ms))

        print("输入 : %s" % np.array2string(host_in, precision=4))
        print("NPU输出: %s" % np.array2string(last, precision=4))

        if args.ref:
            ref = np.load(args.ref).astype(np.float32).ravel()
            print("CPU参考: %s" % np.array2string(ref, precision=4))
            max_diff = float(np.max(np.abs(last - ref)))
            ok = bool(np.allclose(last, ref, rtol=1e-2, atol=1e-2))
            print("最大误差: %.6g，数值比对: %s" % (max_diff, "PASS" if ok else "FAIL"))
            if not ok:
                return 2
    finally:
        acl.rt.free(dev_in)
        acl.rt.free(dev_out)
        acl.mdl.unload(model_id)
        acl.mdl.destroy_desc(desc)
        acl.rt.destroy_context(context)
        acl.rt.reset_device(args.device)
        acl.finalize()

    print("[7] NPU 推理链路验证完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
