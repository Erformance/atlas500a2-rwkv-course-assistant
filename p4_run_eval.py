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
    parser.add_argument("--samples-out", default="",
                        help="把逐题结果写成 JSONL（任务书 Task C 需要逐题 correctness）")
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

    def jsonable(obj):
        """把 lm-eval 配置里的**函数对象**变成字符串。

        9-28 的教训：`results["configs"]` 里含 `process_docs` 之类的可调用对象，
        `json.dump` 会在写到一半时抛 `TypeError: Object of type function is not JSON
        serializable`，留下一个**被截断的 JSON**（fp16 与均衡档都断在同一行）。
        """
        try:
            json.dumps(obj)
            return obj
        except TypeError:
            if isinstance(obj, dict):
                return {k: jsonable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [jsonable(v) for v in obj]
            return repr(obj)

    out = {
        "lm_eval_version": getattr(lm_eval, "__version__", "unknown"),
        "python": platform.python_version(),
        "tasks": tasks,
        "limit": args.limit,
        "base_url": args.base_url,
        "wall_seconds": round(wall, 1),
        "results": results.get("results", {}),
        "n_samples": {k: len(v) for k, v in (results.get("samples") or {}).items()},
        "configs": jsonable({k: v for k, v in (results.get("configs") or {}).items()}),
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2, default=str)

    # 任务书 Task C：逐题结果必须落盘（此前只存了聚合指标，无法做 paired bootstrap / McNemar）
    if args.samples_out:
        samples = results.get("samples") or {}
        n = 0
        with open(args.samples_out, "w", encoding="utf-8") as fh:
            for task_name, rows in samples.items():
                for r in rows:
                    # lm-eval 的逐题字段名随任务类型不同：acc / acc_norm / exact_match
                    fields = {k: v for k, v in r.items()
                              if k in ("doc_id", "target", "filter", "acc", "acc_norm",
                                       "exact_match", "exact_match_stderr", "acc_stderr",
                                       "acc_norm_stderr")}
                    # 目标 token 可能是嵌套结构，展平成字符串便于落盘
                    if isinstance(fields.get("target"), (list, tuple, dict)):
                        fields["target"] = json.dumps(fields["target"], ensure_ascii=False)
                    # 逐题 NLL（LAMBADA 的 paired NLL 需要）
                    for key in ("nll", "nll_stderr", "loglikelihood", "seq_loglikelihood"):
                        if key in r:
                            fields[key] = r[key]
                    fh.write(json.dumps({"task": task_name, **fields},
                                        ensure_ascii=False) + "\n")
                    n += 1
        print("逐题结果已写出 %s（%d 条）" % (args.samples_out, n))

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
