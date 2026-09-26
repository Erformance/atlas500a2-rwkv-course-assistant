#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P4：在设备上跑 lm-evaluation-harness（自定义 backend = 我们的 NPU 引擎）。

用法（lmeval venv）：
    python p4_run_eval.py --tasks piqa --limit 20 --base-url http://127.0.0.1:8100 \
        --out /home/disk/models/rwkv7-2.9b/p4_eval_piqa.json

说明：
  * 只跑要用的任务，`--limit` 控制子集大小（这台设备 prefill 只有 5~12 tok/s，
    完整基准不可行，见报告里的成本测算）；
  * harness 版本、任务版本、子集大小都会写进输出 JSON，便于复现。
"""

import argparse
import json
import os
import platform
import time

import lm_eval
from lm_eval import utils as lm_utils

import p4_lm_eval_backend                      # noqa: F401  （导入即注册模型）


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="piqa",
                        help="逗号分隔，例如 piqa,lambada_openai")
    parser.add_argument("--limit", type=float, default=20)
    parser.add_argument("--base-url", default="http://127.0.0.1:8100")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    t0 = time.time()
    results = lm_eval.simple_evaluate(
        model="rwkv7_npu",
        model_args="base_url=%s,batch_size=%d" % (args.base_url, args.batch_size),
        tasks=tasks,
        limit=args.limit,
        bootstrap_iters=0,                     # 子集太小，不做 bootstrap 置信区间
        log_samples=True,
    )
    wall = time.time() - t0

    out = {
        "lm_eval_version": getattr(lm_eval, "__version__", "unknown"),
        "python": platform.python_version(),
        "tasks": tasks,
        "limit": args.limit,
        "base_url": args.base_url,
        "wall_seconds": round(wall, 1),
        "results": results.get("results", {}),
        "n_samples": {k: len(v) for k, v in (results.get("samples") or {}).items()},
        "configs": {k: v for k, v in (results.get("configs") or {}).items()},
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    print("\n==== 结果（lm-eval %s，子集 %s）===="
          % (out["lm_eval_version"], args.limit))
    for task, res in results.get("results", {}).items():
        main_metric = None
        for key in ("acc,none", "acc_norm,none", "exact_match,strict-match", "acc"):
            if key in res:
                main_metric = (key, res[key])
                break
        print("  %-20s %s" % (task, ("%s = %.4f" % main_metric) if main_metric else res))
    print("用时 %.0f s，已写出 %s" % (wall, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
