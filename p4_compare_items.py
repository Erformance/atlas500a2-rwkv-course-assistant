#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P4 ①：比较 NPU 与 CPU 在同一批题目上的逐题 logp（协议 §6 的"先核对少量逐题结果"）。"""

import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npu", default="p4_npu_scores.json")
    parser.add_argument("--cpu", default="p4_cpu_scores.json")
    parser.add_argument("--out", default="p4_cpu_vs_npu.json")
    args = parser.parse_args()

    with open(args.npu, encoding="utf-8") as fh:
        npu = json.load(fh)
    with open(args.cpu, encoding="utf-8") as fh:
        cpu = json.load(fh)
    by_cpu = {r["index"]: r for r in cpu["rows"]}

    report = {"npu": {k: v for k, v in npu.items() if k != "rows"},
              "cpu": {k: v for k, v in cpu.items() if k != "rows"}, "items": []}
    print("%-6s %-28s %-28s %-8s %s" % ("题号", "NPU logp[选项1,选项2]", "CPU logp[选项1,选项2]", "Δ最大", "选择一致"))
    same_choice = 0
    for r in npu["rows"]:
        c = by_cpu.get(r["index"])
        if not c:
            continue
        deltas = [abs(a - b) for a, b in zip(r["logp"], c["logp"])]
        pick_npu = int(r["logp"][0] < r["logp"][1])
        pick_cpu = int(c["logp"][0] < c["logp"][1])
        same_choice += int(pick_npu == pick_cpu)
        report["items"].append({"index": r["index"], "label": r["label"],
                                "npu_logp": r["logp"], "cpu_logp": c["logp"],
                                "max_abs_delta": round(max(deltas), 4),
                                "same_choice": pick_npu == pick_cpu,
                                "npu_correct": pick_npu == r["label"],
                                "cpu_correct": pick_cpu == r["label"]})
        print("%-6d %-28s %-28s %-8.4f %s"
              % (r["index"], r["logp"], c["logp"], max(deltas),
                 "是" if pick_npu == pick_cpu else "**否**"))
    n = len(report["items"])
    report["same_choice_rate"] = round(same_choice / n, 4) if n else None
    report["max_abs_delta_overall"] = round(max((i["max_abs_delta"] for i in report["items"]),
                                                default=0.0), 4)
    print("\n选择一致 %d/%d｜整体最大 logp 偏差 %.4f"
          % (same_choice, n, report["max_abs_delta_overall"]))
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print("已写出 %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
