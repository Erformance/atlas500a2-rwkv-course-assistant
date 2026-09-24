#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 pareto_p1.json 里的计划落成 rwkv7_chat.py / rwkv7_http.py 能直接用的计划文件。

用法: python make_plans_p1.py --pareto pareto_p1.json --prefix plan_p1_top
输出: plan_p1_top8.json / plan_p1_top16.json …（内容为 {"suffix","layers","head"}）
"""

import argparse
import json
import os

MODEL_DIR = "/home/disk/models/rwkv7-2.9b"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pareto", default=os.path.join(MODEL_DIR, "pareto_p1.json"))
    parser.add_argument("--prefix", default="plan_p1_top")
    args = parser.parse_args()

    with open(args.pareto, encoding="utf-8") as fh:
        data = json.load(fh)
    for name, plan in data["plans"].items():
        item = {"suffix": data.get("suffix", "_p1_q"),
                "layers": plan["layers"], "head": plan["head"]}
        path = os.path.join(MODEL_DIR, "%s%s.json" % (args.prefix, name.replace("top", "")))
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(item, fh, ensure_ascii=False, indent=2)
        print("%-28s int8 %2d 层 + head %-5s KL %-7s top-1 %s"
              % (os.path.basename(path), len(item["layers"]), item["head"],
                 plan.get("kl_mean"), plan.get("top1_agreement")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
