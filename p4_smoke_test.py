#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P4 冒烟：直接打打分服务的两个接口，确认口径与方向正确（不依赖 lm-eval）。

期望：语义正确的续写 logp 明显高于错的；generate_until 能正常续写并在 until 处截断。
用法: python p4_smoke_test.py [--base-url http://127.0.0.1:8100]
"""

import argparse
import json
import time
import urllib.request


def post(base, path, obj, timeout=900):
    req = urllib.request.Request(base.rstrip("/") + path,
                                 data=json.dumps(obj).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8100")
    args = parser.parse_args()

    pairs = [
        ["Question: How do you close a door?\nAnswer:", " Turn the handle and push."],
        ["Question: How do you close a door?\nAnswer:", " boil water in a pot"],
        ["The capital of France is", " Paris"],
        ["The capital of France is", " Tokyo"],
    ]
    t0 = time.time()
    res = post(args.base_url, "/loglikelihood", {"requests": pairs})["results"]
    print("loglikelihood（%.1fs）：" % (time.time() - t0))
    for (ctx, cont), (logp, greedy) in zip(pairs, res):
        print("  %-46s %-24s logp %9.3f  greedy=%s"
              % (ctx[:44].replace("\n", " "), cont[:22], logp, greedy))
    ok = (res[0][0] > res[1][0]) and (res[2][0] > res[3][0])
    print("方向检查（正确续写 logp 更高）：%s" % ("通过" if ok else "**不通过**"))

    t1 = time.time()
    gen = post(args.base_url, "/generate_until",
               {"requests": [["The capital of France is", {"max_gen_toks": 8, "until": ["\n"]}]]})
    print("generate_until（%.1fs）：%s" % (time.time() - t1, json.dumps(gen, ensure_ascii=False)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
