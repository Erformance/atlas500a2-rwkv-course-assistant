#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P3-E：协议 §5 的"额外测量"——稠密 Gramian 与结构化递推的数值一致性 + 离线耗时。

协议原文要求："额外测量：稠密 Gramian 与结构化递推的数值一致性和离线耗时，报告模型大小、
窗口数、head 数、H 和所用计算设备。NumPy 的 O(d²) 代码通过并不表示在 NPU 上一定更快。"

用到的真实递推（取自 `modeling_rwkv7.py` 的 RWKV-7 参考实现）：

    S_t = diag(w_t) S_{t-1} - a_t kk_t^T S_{t-1} + k_t v_t^T ,   y = r_t^T S_t

即每步的状态转移算子是 **对角矩阵 + 秩 1 修正**：M_t = diag(exp(w_log_t)) − a_t kk_tᵀ。
两种算法算同一个对象 G = Σ_t g_t g_tᵀ（g_t = ∂(r_Hᵀ S_H)/∂S_t）：

  * dense：显式构造每个 M_t（d×d），做矩阵乘得到 Jacobian 乘积 P_t，
           再取 g_t = P_t r_H           —— 成本 O(H·d³)
  * structured：反向递推 g_t = M_{t+1}ᵀ g_{t+1}，其中 Mᵀv = w⊙v − kk(a·v)
           —— 成本 O(H·d²)

做法：
  capture  在 CPU 上跑 8 个 token 的**真实**前向，记下每层每 head 的 w_log / a / kk / r
  study    用真实量在 H=8 上比对两种算法（最大偏差、相对偏差）并计时；
           再用循环拼接得到的 H=32/128 窗口报告耗时随 H 的缩放（标注为合成窗口）
输出：p3e_capture.npz、p3e_gramian.json

用法（rwkv7 环境）：
  python p3e_gramian.py --mode capture
  python p3e_gramian.py --mode study
  python p3e_gramian.py
"""

import argparse
import json
import os
import sys
import time

import numpy as np

MODEL_DIR = os.environ.get("RWKV_MODEL_DIR", "/home/disk/models/rwkv7-2.9b")
CAP = os.path.join(MODEL_DIR, "p3e_capture.npz")
OUT = os.path.join(MODEL_DIR, "p3e_gramian.json")
CAPTURE_TOKENS = 8


def capture():
    sys.path.insert(0, MODEL_DIR)
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import modeling_rwkv7 as M

    try:
        torch.backends.mkldnn.enabled = False
    except Exception:
        pass
    torch.set_num_threads(4)

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    import p1_corpus
    ids = tok.encode(p1_corpus.text_for("val"))[:CAPTURE_TOKENS]
    print("捕获序列 %d token" % len(ids), flush=True)

    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, trust_remote_code=True,
                                                 dtype=torch.bfloat16).eval()
    print("模型层数 %d" % len(model.rwkv7.blocks), flush=True)

    store = {}
    orig = M.RWKV7_WKV_FUNCTIONS["eager"]

    def wrapper(r, w_log, k, v, kk, a, state, *args, **kwargs):
        # 记录每层：w_log / a / kk / r（[B,T,H,d]）
        for key, val in (("w_log", w_log), ("a", a), ("kk", kk), ("r", r)):
            store.setdefault(key, []).append(val.detach().float().cpu().numpy())
        return orig(r, w_log, k, v, kk, a, state, *args, **kwargs)

    M.RWKV7_WKV_FUNCTIONS["eager"] = wrapper
    x = torch.tensor([ids], dtype=torch.long)
    t0 = time.time()
    with torch.no_grad():
        model(x)
    print("真实前向 %.0f s" % (time.time() - t0), flush=True)
    M.RWKV7_WKV_FUNCTIONS["eager"] = orig

    layers = len(store["w_log"])
    print("捕获到 %d 层，每层 shape %s" % (layers, store["w_log"][0].shape), flush=True)
    np.savez_compressed(CAP, n_layers=layers, tokens=len(ids),
                        **{k: np.stack(v, axis=0) for k, v in store.items()})
    print("已写出 %s" % os.path.basename(CAP), flush=True)
    return 0


def _window(arr, layer, H):
    """取某一层在窗口 H 上的量：[H, heads, d]（超出捕获步数时循环拼接）。"""
    x = arr[layer][0]                     # [T, heads, d]
    rep = int(np.ceil(H / x.shape[0]))
    return np.concatenate([x] * rep, axis=0)[:H]


def gramian_dense(w_log, a, kk, r, H):
    """显式构造 d×d 转移矩阵并做矩阵乘（O(H·d³)）。"""
    heads, d = w_log.shape[1], w_log.shape[2]
    G = np.zeros((heads, d, d))
    decay = np.exp(np.clip(w_log[:H], -50, 0))
    for h in range(heads):
        Ms = []
        for t in range(H):
            Mt = np.diag(decay[t, h])
            Mt -= np.outer(a[t, h], kk[t, h])
            Ms.append(Mt)
        P = np.eye(d)
        for t in range(H - 1, -1, -1):
            G[h] += np.outer(P.T @ r[t, h], P.T @ r[t, h])
            P = Ms[t] @ P
    return G


def gramian_structured(w_log, a, kk, r, H):
    """反向递推 g_t = Mᵀ g，利用对角+秩1结构（O(H·d²)）。"""
    heads, d = w_log.shape[1], w_log.shape[2]
    G = np.zeros((heads, d, d))
    decay = np.exp(np.clip(w_log[:H], -50, 0))
    for h in range(heads):
        g = r[H - 1, h].copy()
        for t in range(H - 1, -1, -1):
            G[h] += np.outer(g, g)
            if t > 0:
                g = decay[t, h] * g - kk[t, h] * float(a[t, h] @ g)
    return G


def study():
    d = np.load(CAP)
    w_log, a, kk, r = d["w_log"], d["a"], d["kk"], d["r"]
    n_layers, steps, heads, dim = w_log.shape[0], w_log.shape[2], w_log.shape[3], w_log.shape[4]
    print("捕获：%d 层 × %d head × d=%d，%d 步" % (n_layers, heads, dim, steps), flush=True)

    report = {"model": os.path.basename(MODEL_DIR), "layers": int(n_layers),
              "heads": int(heads), "head_dim": int(dim),
              "captured_steps": int(steps), "windows": {}}

    for H in (8, 32, 128):
        if H > steps:
            note = "合成窗口（捕获 %d 步循环拼接）" % steps
        else:
            note = "真实窗口"
        t0 = time.time()
        Gd = np.stack([gramian_dense(_window(w_log, L, H), _window(a, L, H),
                                     _window(kk, L, H), _window(r, L, H), H)
                       for L in range(n_layers)])
        t_dense = time.time() - t0

        t0 = time.time()
        Gs = np.stack([gramian_structured(_window(w_log, L, H), _window(a, L, H),
                                          _window(kk, L, H), _window(r, L, H), H)
                       for L in range(n_layers)])
        t_struct = time.time() - t0

        denom = np.abs(Gd).max() + 1e-30
        max_abs = float(np.abs(Gd - Gs).max())
        rel = float(np.abs(Gd - Gs).max() / denom)
        report["windows"]["H%d" % H] = {
            "note": note, "max_abs_diff": max_abs, "rel_diff": rel,
            "dense_seconds": round(t_dense, 3), "structured_seconds": round(t_struct, 3),
            "dense_over_structured": round(t_dense / max(t_struct, 1e-9), 2)}
        print("H=%-4d %-22s 最大偏差 %.3e（相对 %.2e）｜ dense %.2fs ｜ structured %.2fs ｜ 倍数 %.1f×"
              % (H, note, max_abs, rel, t_dense, t_struct,
                 t_dense / max(t_struct, 1e-9)), flush=True)

    with open(OUT, "w") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print("\n已写出 %s" % os.path.basename(OUT), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="all", choices=["all", "capture", "study"])
    args = parser.parse_args()
    if args.mode in ("all", "capture"):
        capture()
    if args.mode in ("all", "study"):
        study()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
