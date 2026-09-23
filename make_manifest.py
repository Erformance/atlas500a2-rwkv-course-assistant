#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""采集补实验协议第 1 节要求的 manifest.json。

在设备上运行；git_commit 由外部传入（设备上没有仓库）。

用法: python make_manifest.py --git-commit <sha> [--git-dirty true]
输出: /home/disk/models/rwkv7-2.9b/manifest.json
"""

import argparse
import glob
import hashlib
import json
import os
import platform
import subprocess

MODEL_DIR = "/home/disk/models/rwkv7-2.9b"


def sh(cmd, default=""):
    try:
        out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        return (out.stdout or out.stderr).strip()
    except Exception as exc:
        return "%s: %s" % (default, exc)


def sha256(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
            if limit and fh.tell() > limit:
                break
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--git-commit", default="")
    parser.add_argument("--git-dirty", default="unknown")
    parser.add_argument("--out", default=os.path.join(MODEL_DIR, "manifest.json"))
    args = parser.parse_args()

    # ---- 模型 ----
    cfg_path = os.path.join(MODEL_DIR, "config.json")
    config = {}
    if os.path.exists(cfg_path):
        with open(cfg_path) as fh:
            config = json.load(fh)
    shards = sorted(glob.glob(os.path.join(MODEL_DIR, "model-*.safetensors")))
    weights = {os.path.basename(p): {"size": os.path.getsize(p), "sha256": sha256(p)}
               for p in shards}
    tok_path = os.path.join(MODEL_DIR, "tokenizer.json")

    # ---- 编译产物（只记大小；om 的 sha256 逐个算太慢，按需再算） ----
    oms = {}
    for name in ["head.om"] + ["layer%d.om" % i for i in range(32)]:
        p = os.path.join(MODEL_DIR, name)
        if os.path.exists(p):
            oms[name] = os.path.getsize(p)

    # ---- 硬件 ----
    hardware = {
        "npu_smi": sh("npu-smi info | sed -n '6,8p'"),
        "cpu": platform.machine(),
        "mem_total_kb": sh("awk '/MemTotal/{print $2}' /proc/meminfo"),
        "mem_available_kb": sh("awk '/MemAvailable/{print $2}' /proc/meminfo"),
        "os": sh("cat /etc/os-release | tr '\\n' ' '"),
        "kernel": platform.release(),
    }

    # ---- 软件版本 ----
    soft = {
        "cann_version": sh("cat /home/disk/cann80base/ascend-toolkit/latest/version.cfg 2>/dev/null | head -3"),
        "nnrt": sh("ls /usr/local/Ascend/nnrt 2>/dev/null | head -3"),
        "atc": sh("source /home/disk/cann80base/ascend-toolkit/set_env.sh >/dev/null 2>&1; "
                  "export PATH=/home/disk/miniconda3/envs/npu22/bin:$PATH; atc --version 2>&1 | head -2"),
        "python": sh("/home/disk/miniconda3/envs/npu22/bin/python -V"),
        "torch_export_env_python": sh("/home/disk/miniconda3/envs/rwkv7/bin/python -V"),
        "acl_python": sh("/home/disk/miniconda3/envs/npu22/bin/python -c"
                         " 'import acl; print(acl.__file__)' 2>&1 | tail -1"),
        "modelslim": sh("ls /home/disk/cann80base/ascend-toolkit/latest/tools/modelslim 2>/dev/null | head -3"),
    }

    manifest = {
        "schema": "atlas500a2-rwkv-quant-manifest/1",
        "created": sh("date -u +%Y-%m-%dT%H:%M:%SZ"),
        "model_name": "RWKV-7 G1 2.9B (base, 非 instruct)",
        "model_dir": MODEL_DIR,
        "model_revision": {"config_sha256": sha256(cfg_path) if os.path.exists(cfg_path) else None,
                           "config": config},
        "tokenizer_hash": sha256(tok_path) if os.path.exists(tok_path) else None,
        "weight_files": weights,
        "om_files": oms,
        "git_commit": args.git_commit,
        "git_dirty": args.git_dirty,
        "hardware": hardware,
        "software_versions": soft,
        "compile_commands": {
            "fp16_layer": ("atc --model=layerN.onnx --framework=5 --output=layerN "
                           "--soc_version=Ascend310B1 --input_format=ND --log=error"),
            "int8_layer": ("atc --model=layerN_q.onnx --framework=5 --output=layerN_q "
                           "--soc_version=Ascend310B1 --input_format=ND --log=error"),
            "export": ("python rwkv7_export_chunk.py --start N --layers 1 --dtype float32 "
                       "--out layerN.onnx  (opset 14, do_constant_folding=False)"),
        },
        "candidate_plan": ["fp16", "int8-label-free", "int8-calibrated(pre-exec sampling)", "ffn-only-int8"],
        "calibration_ids": {"pilot": "calib_pre/layerNN.npz (16 steps, 状态年龄 0~15)",
                            "design": "P1 计划：16→128 段短文，状态年龄 0/64/256(/1024) 分层采样"},
        "validation_ids": "待建（与 calibration/test 分离）",
        "test_ids": "待建（不得用于调 scale/阈值/提示词）",
        "seeds": {"sampling": 0, "note": "贪心解码 temperature=0"},
        "generation_settings": config.get("dtype", "bfloat16") + " 原始权重，推理 fp16；max_tokens 由调用方给定",
    }
    with open(args.out, "w") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    print(json.dumps({k: manifest[k] for k in
                      ("created", "git_commit", "tokenizer_hash", "hardware")},
                     ensure_ascii=False, indent=2))
    print("\n已写出 %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
