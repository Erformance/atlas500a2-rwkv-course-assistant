#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P5 固定工作量微基准（协议 §7）。

对每个精度臂 × 每个前缀长度，测：
  * cold load：引擎构造耗时（.om 加载 + embedding 载入）
  * prefill：零状态喂 N 个**预定** token 的耗时（不含加载）
  * TTFT：前缀喂完到吐出第一个 token（含采样/argmax）
  * decode：128 个 token 的**纯解码**耗时与吞吐（不含加载与 prefill）
  * 内存：进程 RSS、系统可用内存差值、共享内存差值；权重/状态/embedding 由常量算出
  * 温度：每个 repeat 前后的 npu-smi 温度

每个 (臂, repeat) 跑在**独立子进程**里（同进程反复建 Engine 会造成设备内存不释放），
重复之间随机化前缀顺序；结果写 p5_parts/ 可续跑，汇总到 p5_bench.json。

注意：这里刻意不用"内容触发的提前停止"，全部按预定 token 驱动——测的是执行性能，不是质量。

用法:
  python p5_bench.py                       # 全流程（4 臂 × 5 次重复）
  python p5_bench.py --arms precise,balanced --repeats 2
"""

import argparse
import json
import os
import random
import resource
import subprocess
import sys
import time

import numpy as np

MODEL_DIR = "/home/disk/models/rwkv7-2.9b"
PY = "/home/disk/miniconda3/envs/npu22/bin/python"
EMBED_BYTES = 65536 * 2560 * 4          # CPU 端 embedding（65536×2560 fp32）
STATE_BYTES_PER_LAYER = 2 * 2560 * 4 + 40 * 64 * 64 * 4

ARMS = {
    "precise":   {"suffix": "", "plan": None, "label": "精确档 fp16"},
    "balanced":  {"plan": "plan_p1_top8.json", "label": "均衡档 p1 top8"},
    "fast":      {"plan": "plan_p1_top16.json", "label": "快速档 p1 top16"},
    "full_int8": {"plan": "plan_p1_top33.json", "label": "全量 int8（实验）"},
}


def meminfo():
    out = {}
    with open("/proc/meminfo") as fh:
        for line in fh:
            k, _, v = line.partition(":")
            out[k] = int(v.strip().split()[0])      # kB
    return out


def snd_temp():
    try:
        r = subprocess.run(["npu-smi", "info"], capture_output=True, text=True)
        for line in r.stdout.splitlines():
            if "310B" in line:
                for tok in line.split("|"):
                    tok = tok.strip()
                    if tok.isdigit() and 20 <= int(tok) <= 90:
                        return int(tok)
    except Exception:
        pass
    return None


def worker(arm, repeat, prefixes, decode_tokens, ids_all):
    sys.path.insert(0, MODEL_DIR)
    from rwkv7_serve2 import Engine

    cfg = ARMS[arm]
    plan = None
    if cfg.get("plan"):
        with open(os.path.join(MODEL_DIR, cfg["plan"])) as fh:
            plan = json.load(fh)
    m0 = meminfo()
    t0 = time.time()
    engine = Engine(MODEL_DIR, suffix=cfg.get("suffix", ""), plan=plan)
    load_s = time.time() - t0
    m1 = meminfo()
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    order = list(prefixes)
    random.Random(repeat).shuffle(order)        # 重复之间随机化顺序
    rows = {}
    temp0 = snd_temp()
    for n in order:
        ids = ids_all[:n]
        fed = len(ids)
        engine.reset_state()
        t1 = time.time()
        logits = None
        for tok in ids:
            logits = engine.step(int(tok))
        prefill_s = time.time() - t1
        # TTFT：前缀已喂完，第一个 token 的采样 + 一步
        t2 = time.time()
        nxt = int(np.argmax(logits))
        logits = engine.step(nxt)
        ttft_s = prefill_s + (time.time() - t2)
        # 纯解码：再来 decode_tokens 步，只计时 step() 循环（不含 argmax）
        t3 = time.time()
        for _ in range(decode_tokens):
            logits = engine.step(int(np.argmax(logits)))
        decode_s = time.time() - t3
        rows[str(n)] = {
            "n_tokens_fed": fed,
            "prefill_s": round(prefill_s, 3),
            "prefill_tok_s": round(fed / prefill_s, 2) if prefill_s else None,
            "ttft_ms": round(ttft_s * 1000, 1),
            "decode_s": round(decode_s, 3),
            "decode_ms_per_token": round(decode_s / decode_tokens * 1000, 2),
            "decode_tok_s": round(decode_tokens / decode_s, 2) if decode_s else None,
        }
        print("  %s r%d prefix=%-5d prefill %7.3fs  ttft %7.1fms  decode %6.2f tok/s"
              % (arm, repeat, n, prefill_s, ttft_s * 1000,
                 rows[str(n)]["decode_tok_s"] or 0), flush=True)
    temp1 = snd_temp()
    m2 = meminfo()
    del engine
    return {
        "arm": arm, "repeat": repeat, "order": order,
        "load_s": round(load_s, 2), "n_int8_layers": None,
        "rss_mb_after_load": round(rss_mb, 1),
        "mem_available_delta_mb": round((m0["MemAvailable"] - m1["MemAvailable"]) / 1024.0, 1),
        "shmem_delta_mb": round((m2["Shmem"] - m0["Shmem"]) / 1024.0, 1),
        "temp_c": {"before": temp0, "after": temp1},
        "prefixes": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", default="precise,balanced,fast,full_int8")
    parser.add_argument("--prefixes", default="32,128,512,1024,2048")
    parser.add_argument("--decode", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5,
                        help="重复次数；2048 前缀自动只跑 3 次（单次 prefill 就要几分钟）")
    parser.add_argument("--worker-arm", default=None)
    parser.add_argument("--worker-repeat", type=int, default=None)
    parser.add_argument("--corpus-repeat", type=int, default=4,
                        help="PILOT_TEXT 重复次数（决定可用最长前缀；4 次≈1100 token）")
    parser.add_argument("--out", default=os.path.join(MODEL_DIR, "p5_bench.json"))
    parser.add_argument("--parts-dir", default=os.path.join(MODEL_DIR, "p5_parts"))
    args = parser.parse_args()
    prefixes = [int(x) for x in args.prefixes.split(",")]

    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
    from capture_calib_pre import PILOT_TEXT
    ids_all = tokenizer.encode(PILOT_TEXT * args.corpus_repeat).ids
    print("语料可用 token 数：%d（前缀上限受此限制）" % len(ids_all), flush=True)

    # ---- 子进程模式 ---------------------------------------------------------
    if args.worker_arm:
        row = worker(args.worker_arm, args.worker_repeat, prefixes, args.decode, ids_all)
        with open(os.path.join(args.parts_dir,
                               "%s_r%d.json" % (args.worker_arm, args.worker_repeat)), "w") as fh:
            json.dump(row, fh, ensure_ascii=False, indent=2)
        print(json.dumps({k: v for k, v in row.items() if k != "prefixes"}, ensure_ascii=False))
        return 0

    # ---- 驱动模式 -----------------------------------------------------------
    os.makedirs(args.parts_dir, exist_ok=True)
    results = {}
    for arm in args.arms.split(","):
        for r in range(1, args.repeats + 1):
            part = os.path.join(args.parts_dir, "%s_r%d.json" % (arm, r))
            if not os.path.exists(part):
                # 长前缀（>1024）自动只跑前 3 次：单次 2048 前缀的 prefill 就要 6 分钟左右，
                # 在 11.5GB 的板子上没必要为它堆满 5 次（协议也允许样本不足时只报原始样本与中位数）
                prefs = [p for p in prefixes if p <= 1024 or r <= 3]
                cmd = [PY, os.path.join(MODEL_DIR, "p5_bench.py"),
                       "--prefixes", ",".join(str(x) for x in prefs), "--decode", str(args.decode),
                       "--corpus-repeat", str(args.corpus_repeat),
                       "--worker-arm", arm, "--worker-repeat", str(r),
                       "--parts-dir", args.parts_dir]
                t0 = time.time()
                p = subprocess.run(cmd, capture_output=True, text=True)
                print(p.stdout.strip()[-300:] or p.stderr.strip()[-300:], flush=True)
                print("  %s r%d 用时 %.0f s" % (arm, r, time.time() - t0), flush=True)
            if os.path.exists(part):
                with open(part) as fh:
                    results.setdefault(arm, []).append(json.load(fh))

    # 汇总：每个 (臂, 前缀) 取中位数
    summary = {"configs": {"prefixes": prefixes, "decode_tokens": args.decode,
                           "repeats": args.repeats,
                           "embed_bytes": EMBED_BYTES,
                           "state_bytes": STATE_BYTES_PER_LAYER * 32},
               "arms": {}}
    for arm, rows in results.items():
        entry = {"n_repeats": len(rows),
                 "load_s_median": round(float(np.median([r["load_s"] for r in rows])), 2),
                 "rss_mb_median": round(float(np.median([r["rss_mb_after_load"] for r in rows])), 1),
                 "mem_available_delta_mb_median": round(float(np.median(
                     [r["mem_available_delta_mb"] for r in rows])), 1),
                 "prefixes": {}}
        for n in prefixes:
            vals = [r["prefixes"][str(n)] for r in rows if str(n) in r["prefixes"]]
            if not vals:
                continue
            entry["prefixes"][str(n)] = {
                "n": len(vals),
                "prefill_s_median": round(float(np.median([v["prefill_s"] for v in vals])), 3),
                "ttft_ms_median": round(float(np.median([v["ttft_ms"] for v in vals])), 1),
                "decode_tok_s_median": round(float(np.median([v["decode_tok_s"] for v in vals])), 2),
                "decode_ms_per_token_median": round(float(np.median(
                    [v["decode_ms_per_token"] for v in vals])), 2),
            }
        summary["arms"][arm] = entry
        p = entry["prefixes"]
        print("\n%-10s 加载 %5.1fs｜RSS %7.1fMB｜可用内存差 %7.1fMB" %
              (arm, entry["load_s_median"], entry["rss_mb_median"],
               entry["mem_available_delta_mb_median"]))
        for n in prefixes:
            if str(n) in p:
                print("   prefix %-5d prefill %7.3fs｜TTFT %8.1fms｜decode %6.2f tok/s（%5.2f ms/token）"
                      % (n, p[str(n)]["prefill_s_median"], p[str(n)]["ttft_ms_median"],
                         p[str(n)]["decode_tok_s_median"],
                         p[str(n)]["decode_ms_per_token_median"]))
    with open(args.out, "w") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print("\n已写出 %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
