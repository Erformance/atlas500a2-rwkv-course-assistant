#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P3-D 前置探测：确认既有臂数据口径与各文本的可用 token 长度。

回答三个问题：
  1) 既有 arm_fp16*.npz 是几步（H=64 还是别的）？
  2) pilot / mixed / p1val / p1test 四份文本各有多少 token（H=128 需要 >=128）？
  3) p3_result.json 里记录的 H 与参考轨迹步数。
"""

import json
import os
import sys

import numpy as np

MODEL_DIR = os.environ.get("RWKV_MODEL_DIR", "/home/disk/models/rwkv7-2.9b")
sys.path.insert(0, MODEL_DIR)

from tokenizers import Tokenizer                      # noqa: E402
from capture_calib_pre import PILOT_TEXT, MIXED_TEXT  # noqa: E402
import p1_corpus                                      # noqa: E402


def main():
    print("=== 既有臂数据 ===")
    for name in sorted(os.listdir(MODEL_DIR)):
        if name.startswith("arm_") and name.endswith(".npz"):
            path = os.path.join(MODEL_DIR, name)
            try:
                d = np.load(path)
                lg = d["logits"]
                print("%-28s steps=%-5d vocab=%-6d keys=%s"
                      % (name, lg.shape[0], lg.shape[1], ",".join(d.files)))
            except Exception as exc:                  # noqa: BLE001
                print("%-28s 读取失败: %s" % (name, exc))

    print("\n=== 文本 token 长度 ===")
    tok = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
    texts = {"pilot": PILOT_TEXT, "mixed": MIXED_TEXT,
             "p1val": p1_corpus.text_for("val"), "p1test": p1_corpus.text_for("test")}
    for name, text in texts.items():
        n = len(tok.encode(text).ids)
        print("%-8s tokens=%d" % (name, n))

    print("\n=== p3_result.json ===")
    path = os.path.join(MODEL_DIR, "p3_result.json")
    if os.path.exists(path):
        with open(path) as fh:
            rep = json.load(fh)
        print("targets=%s horizons=%s" % (rep.get("targets"), rep.get("horizons")))
        print("spearman_vs_h64=%s" % rep.get("spearman_vs_h64"))
        print("bootstrap=%s" % rep.get("bootstrap_spearman"))
    else:
        print("不存在")

    print("\n=== p3_parts 清单 ===")
    parts = os.path.join(MODEL_DIR, "p3_parts")
    files = sorted(os.listdir(parts)) if os.path.isdir(parts) else []
    print("共 %d 个：%s" % (len(files), " ".join(files[:40])))
    if files:
        d = np.load(os.path.join(parts, files[0]))
        print("示例 %s：logits=%s keys=%s"
              % (files[0], d["logits"].shape, ",".join(d.files)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
