#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P3 方向指标第二步：在 PyTorch 参考上算"输出加权方向灵敏度" γᵢ = |gᵢ · δᵢ|。

协议 §5 要求比较"输出加权 Gramian"与局部误差/短 rollout 的预测力。这里实现它的一阶形式：

    gᵢ = ∂(−log p(真实下一个 token)) / ∂(第 i 层块的输出)      ← 一次反向传播同时得到所有层
    δᵢ = int8 层与 fp16 层在同一份真实输入下的输出误差向量     ← 由 p3b_err_vectors.py 在 NPU 上实测
    γᵢ = |gᵢ · δᵢ|

与只看大小的局部指标相比，γᵢ 考虑了**误差方向**与输出敏感度的对齐关系。
注意：CPU 上跑 2.9B 的 bf16 前向+反向很慢（MKLDNN 不支持 bf16，回退实现约 20~30 s/token），
所以只取很短的序列（默认 4 个 token），这也是协议允许的"先做小模型/少量窗口"。

用法（rwkv7 环境，需先有 p3b_err_vec/）：
  python p3b_dir_metric.py --tokens 4 --out p3b_dir.json
"""

import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np

MODEL_DIR = os.environ.get("RWKV_MODEL_DIR", "/home/disk/models/rwkv7-2.9b")
sys.path.insert(0, MODEL_DIR)


def norm(name):
    return re.sub(r"\.\d+$", "", name.split(":")[-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=4, help="用于反传的序列长度（很短，CPU 慢）")
    parser.add_argument("--text", default="val", choices=["val", "test", "pilot"],
                        help="取哪段文本（val/test 来自 p1_corpus，与校准不重叠）")
    parser.add_argument("--err-dir", default=os.path.join(MODEL_DIR, "p3b_err_vec"))
    parser.add_argument("--out", default=os.path.join(MODEL_DIR, "p3b_dir.json"))
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        torch.backends.mkldnn.enabled = False        # aarch64 上 MKLDNN 不支持 bf16 matmul
    except Exception:
        pass
    torch.set_num_threads(4)

    # 1) 取一段真实文本（与校准分割不重叠）
    if args.text == "pilot":
        from capture_calib_pre import PILOT_TEXT as TEXT
    else:
        import p1_corpus
        TEXT = p1_corpus.text_for(args.text)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    ids = tok.encode(TEXT)[: args.tokens + 1]
    print("序列 %d token：%s…" % (len(ids), tok.decode(ids)[:40].replace("\n", " ")), flush=True)

    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, trust_remote_code=True,
                                                 dtype=torch.bfloat16).eval()
    blocks = model.rwkv7.blocks
    n_layer = len(blocks)
    print("模型层数 %d（%s）" % (n_layer, MODEL_DIR), flush=True)

    # 2) 抓每层块输出
    captured = {}

    def make_hook(i):
        def hook(_module, _inputs, output):
            h = output[0] if isinstance(output, (tuple, list)) else output
            captured[i] = h
        return hook

    for i, blk in enumerate(blocks):
        blk.register_forward_hook(make_hook(i))

    x = torch.tensor([ids], dtype=torch.long)
    t0 = time.time()
    with torch.no_grad():
        model(x[:, :-1])                       # 预热（不可反传）
    print("预热一次前向 %.0f s" % (time.time() - t0), flush=True)

    t0 = time.time()
    out = model(x[:, :-1])                     # 前 n-1 个 token 作为输入
    logits = out.logits[0, -1].float()         # 最后一个位置的 logits
    target = int(ids[-1])
    obj = -torch.log_softmax(logits, dim=-1)[target]
    grads = torch.autograd.grad(obj, [captured[i] for i in range(n_layer)],
                                retain_graph=False, allow_unused=True)
    print("前向+反向 %.0f s" % (time.time() - t0), flush=True)

    # 3) 与实测误差向量做内积
    rows = {}
    for i in range(n_layer):
        g = grads[i]
        if g is None:
            rows[i] = {"dir_score": None, "g_norm": None, "delta_norm": None, "cos": None}
            continue
        # 只用**最后一个时间步**的梯度：损失取在最后一位，那里的梯度与
        # "该层输出被扰动后对最终 logits 的影响"最直接对应；δᵢ 是单步的输出误差向量（H 维）。
        gg = g.detach().float()
        gv = (gg[0, -1, :] if gg.dim() == 3 else gg.reshape(-1)).numpy()
        path = os.path.join(args.err_dir, "layer%02d.npz" % i)
        if not os.path.exists(path):
            rows[i] = {"dir_score": None, "g_norm": float(np.linalg.norm(gv)),
                       "delta_norm": None, "cos": None}
            continue
        err = np.load(path)
        key = "x_0" if "x_0" in err.files else sorted(err.files)[0]
        dv = err[key].astype(np.float64).reshape(-1)
        n = min(len(gv), len(dv))
        gv2, dv2 = gv[:n], dv[:n]
        dot = float(np.dot(gv2, dv2))
        rows[i] = {"dir_score": round(abs(dot), 6),
                   "g_norm": round(float(np.linalg.norm(gv2)), 6),
                   "delta_norm": round(float(np.linalg.norm(dv2)), 6),
                   "cos": round(dot / (np.linalg.norm(gv2) * np.linalg.norm(dv2) + 1e-12), 4),
                   "err_key": key}
    report = {"model_dir": MODEL_DIR, "tokens": len(ids), "text": args.text,
              "target_token": target, "layers": rows}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    print("\n%-6s %-12s %-12s %-12s %s" % ("层", "|gᵢ·δᵢ|", "‖gᵢ‖", "‖δᵢ‖", "cos"))
    for i in list(rows)[:6] + list(rows)[-4:]:
        r = rows[i]
        print("%-6d %-12s %-12s %-12s %s"
              % (i, r["dir_score"], r["g_norm"], r["delta_norm"], r["cos"]))
    print("已写出 %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
