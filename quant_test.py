#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 CANN 自带的 modelslim 对一个层 ONNX 做免校准量化（label-free PTQ），验证可行性。

用法: python quant_test.py <输入.onnx> <输出.onnx>
"""

import collections
import os
import sys
import time

sys.path.insert(0, "/home/disk/cann80base/ascend-toolkit/latest/tools")

from modelslim.onnx.post_training_quant import run_quantize, QuantConfig  # noqa: E402


def main():
    src, dst = sys.argv[1], sys.argv[2]
    # is_optimize_graph=False：跳过 modelslim 的图优化/opset 转换，
    # 否则它会把 opset14 的 Unsqueeze(axes 作为输入) 转成非法图
    config = QuantConfig(is_optimize_graph=False)
    print("开始量化 %s" % src, flush=True)
    t0 = time.time()
    run_quantize(src, dst, config)
    print("量化完成，用时 %.1f s" % (time.time() - t0), flush=True)
    print("大小: %.1f MB → %.1f MB" % (os.path.getsize(src) / 1e6, os.path.getsize(dst) / 1e6))

    import onnx
    model = onnx.load(dst, load_external_data=False)
    counter = collections.Counter(n.op_type for n in model.graph.node)
    print("量化后算子统计（前 10）:", dict(counter.most_common(10)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
