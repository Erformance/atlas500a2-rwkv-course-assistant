#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""下载 RWKV7-G1j-1.5B-20260831（与 2.9B 同批次），供"跨规模"实验用。

源：HF `RWKV/RWKV7-G1j-1.5B-20260831`（Apache-2.0），走 hf-mirror。
只拉必需文件（safetensors + config + tokenizer + 自定义建模代码 + LICENSE/README），
目标目录 /home/disk/models/rwkv7-1.5b。

用法: HF_ENDPOINT=https://hf-mirror.com python p4b_download_15b.py
"""

import os
import sys

REPO = "RWKV/RWKV7-G1j-1.5B-20260831"
DEST = "/home/disk/models/rwkv7-1.5b"


def main():
    from huggingface_hub import snapshot_download
    os.makedirs(DEST, exist_ok=True)
    path = snapshot_download(
        repo_id=REPO,
        local_dir=DEST,
        allow_patterns=["*.safetensors", "*.json", "*.py", "*.jinja", "*.txt",
                        "LICENSE", "README.md", ".gitattributes"],
        max_workers=4,
    )
    total = sum(os.path.getsize(os.path.join(dp, f))
                for dp, _, fs in os.walk(path) for f in fs)
    print("已下载到 %s（%.2f GB）" % (path, total / 1e9))
    with open(os.path.join(DEST, "config.json")) as fh:
        print("config:", fh.read()[:600])
    return 0


if __name__ == "__main__":
    sys.exit(main())
