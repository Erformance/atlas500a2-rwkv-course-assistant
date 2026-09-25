#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按协议 §1 回填 manifest：数据分割 ID/哈希、候选计划、候选 OM 哈希。

原 manifest（09-22）里 validation_ids / test_ids 还是"待建"，也没有逐候选的 OM 哈希；
P1 之后这些都有了数据来源，这里一次性补齐，并保留原文件为 manifest.json.bak。

用法: python update_manifest_p1.py
输出: manifest.json（就地更新，先备份）
"""

import glob
import hashlib
import json
import os
import time

MODEL_DIR = "/home/disk/models/rwkv7-2.9b"


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def main():
    mpath = os.path.join(MODEL_DIR, "manifest.json")
    with open(mpath, encoding="utf-8") as fh:
        man = json.load(fh)

    # 1) 数据分割（P1 的 64 段语料：校准 32 / 验证 16 / 测试 16）
    cpath = os.path.join(MODEL_DIR, "p1_corpus_manifest.json")
    if os.path.exists(cpath):
        with open(cpath, encoding="utf-8") as fh:
            corpus = json.load(fh)
        per_split = {}
        for seg in corpus["segments"]:
            per_split.setdefault(seg["split"], []).append(seg["id"])
        man["calibration_ids"] = {
            "pilot": man.get("calibration_ids", {}).get("pilot"),
            "p1_calib": per_split.get("calib", []),
            "p1_calib_sha1": corpus["splits"]["calib"]["sha1"],
        }
        man["validation_ids"] = {
            "p1_val": per_split.get("val", []),
            "p1_val_sha1": corpus["splits"]["val"]["sha1"],
        }
        man["test_ids"] = {
            "p1_test": per_split.get("test", []),
            "p1_test_sha1": corpus["splits"]["test"]["sha1"],
        }
        man["corpus"] = {"file": "p1_corpus_manifest.json",
                         "version": corpus.get("version"),
                         "n_segments": corpus.get("n_segments")}

    # 2) 候选计划 + 计划引用的 OM 哈希
    plans = {}
    for path in sorted(glob.glob(os.path.join(MODEL_DIR, "plan_*.json"))):
        name = os.path.basename(path)
        try:
            with open(path, encoding="utf-8") as fh:
                plan = json.load(fh)
        except Exception as exc:
            plans[name] = {"error": repr(exc)}
            continue
        plans[name] = {"suffix": plan.get("suffix", ""),
                       "layers": plan.get("layers", []),
                       "head": bool(plan.get("head", False))}
    for key, item in plans.items():
        if "error" not in item:
            continue
        print("[警告] %s 解析失败：%s" % (key, item["error"]))
    man["plan_files"] = plans

    needed = set()
    for item in plans.values():
        if "error" in item:
            continue
        suffix = item["suffix"] or ""
        for i in range(32):
            if not suffix or i in item["layers"]:
                needed.add("layer%d%s.om" % (i, suffix))
        if item["head"]:
            needed.add("head%s.om" % suffix)
    # 线上三档（tiers.json）用到的也一并登记
    tpath = os.path.join(MODEL_DIR, "tiers.json")
    with open(tpath, encoding="utf-8") as fh:
        tiers = json.load(fh)
    for key, item in tiers.items():
        if not item.get("plan"):
            continue
        with open(os.path.join(MODEL_DIR, item["plan"]), encoding="utf-8") as fh:
            plan = json.load(fh)
        suffix = plan.get("suffix", "")
        for i in plan.get("layers", []):
            needed.add("layer%d%s.om" % (i, suffix))
        if plan.get("head"):
            needed.add("head%s.om" % suffix)

    om_hashes, missing = {}, []
    t0 = time.time()
    for name in sorted(needed):
        path = os.path.join(MODEL_DIR, name)
        if not os.path.exists(path):
            missing.append(name)
            continue
        om_hashes[name] = {"sha256": sha256(path),
                           "size": os.path.getsize(path)}
    man["om_hashes"] = om_hashes
    man["om_hashes_missing"] = missing
    man["tiers"] = tiers
    man["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    os.replace(mpath, mpath + ".bak")
    with open(mpath, "w", encoding="utf-8") as fh:
        json.dump(man, fh, ensure_ascii=False, indent=2)
    print("manifest 已更新：候选 OM %d 个（缺 %d），计划 %d 份，用时 %.0fs"
          % (len(om_hashes), len(missing), len(plans), time.time() - t0))
    if missing:
        print("缺失（计划引用了但没编译）：%s" % missing[:12])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
