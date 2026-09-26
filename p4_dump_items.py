#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P4 ①：导出 PIQA 的少量题目（与 lm-eval 完全相同的 prompt 模板），供 CPU↔NPU 逐题对照。

模板取自装好的 harness：
    doc_to_text: "Question: {{goal}}\nAnswer:"
    doc_to_choice: [sol1, sol2]（harness 会在前面加一个空格作为 target_delimiter）

用法（lmeval venv）：
    HF_HOME=/home/disk/hf_cache HF_ENDPOINT=https://hf-mirror.com \
      /home/disk/lmeval/bin/python p4_dump_items.py --items 3 --out p4_items.json
"""

import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=int, default=3)
    parser.add_argument("--out", default="p4_items.json")
    args = parser.parse_args()

    from datasets import load_dataset
    ds = load_dataset("baber/piqa", split="validation")
    out = []
    for i in range(args.items):
        d = ds[i]
        out.append({
            "index": i,
            "ctx": "Question: %s\nAnswer:" % d["goal"],
            "choices": [" " + d["sol1"], " " + d["sol2"]],
            "label": int(d["label"]),
        })
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"task": "piqa", "template": "Question: {{goal}}\\nAnswer:",
                   "items": out}, fh, ensure_ascii=False, indent=2)
    print("已写出 %s（%d 题）：" % (args.out, len(out)))
    for it in out:
        print("  #%d label=%d｜%s" % (it["index"], it["label"], it["ctx"].replace("\n", " ")[:70]))
        for c in it["choices"]:
            print("       候选:%s" % c[:60])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
