#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RWKV-7 2.9B 在 Atlas 500 A2（310B）上的 NPU 推理编排器。

结构：embedding 查表（host 侧）→ 32 个层 .om 顺序执行（逐层传 state）→ head .om → logits。
只有 5 个张量在 host/NPU 间搬运，每层约 700KB，带宽压力可忽略。

用法：
  # 校验：与 CPU 参考 logits 比对
  python rwkv7_acl_run.py --verify
  # 生成
  python rwkv7_acl_run.py --prompt "你好" --tokens 16
"""

import argparse
import json
import os
import re
import time

import numpy as np

HIDDEN, HEADS, HEAD_DIM, LAYERS, VOCAB = 2560, 40, 64, 32, 65536
ACL_H2D, ACL_D2H = 1, 2


def norm(name):
    """归一化 ATC 改写过的张量名：
    '/Add_1:0:x' -> 'x'；'x.1' -> 'x'；'v_first.1' -> 'v_first'。
    """
    short = name.split(":")[-1]
    return re.sub(r"\.\d+$", "", short)


def match_name(names, key):
    """在 names 里找出与语义 key 对应的那个名字。"""
    for name in names:
        if name == key or norm(name) == key:
            return name
    for name in names:
        if name.endswith(key) or key in norm(name):
            return name
    raise KeyError("找不到张量 %r，现有: %s" % (key, list(names)))


class OmModel:
    def __init__(self, acl, path):
        self.acl = acl
        self.path = path
        self.model_id, ret = acl.mdl.load_from_file(path)
        self.check(ret, "load %s" % os.path.basename(path))
        self.desc = acl.mdl.create_desc()
        self.check(acl.mdl.get_desc(self.desc, self.model_id), "get_desc")

        n_in = acl.mdl.get_num_inputs(self.desc)
        n_out = acl.mdl.get_num_outputs(self.desc)
        self.in_names = [acl.mdl.get_input_name_by_index(self.desc, i) for i in range(n_in)]
        self.out_names = [acl.mdl.get_output_name_by_index(self.desc, i) for i in range(n_out)]
        self.in_sizes = [acl.mdl.get_input_size_by_index(self.desc, i) for i in range(n_in)]
        self.out_sizes = [acl.mdl.get_output_size_by_index(self.desc, i) for i in range(n_out)]

        self.dev_in, self.dev_out = {}, {}
        for name, size in zip(self.in_names, self.in_sizes):
            ptr, ret = acl.rt.malloc(size, 0)
            self.check(ret, "malloc in")
            self.dev_in[name] = ptr
        for name, size in zip(self.out_names, self.out_sizes):
            ptr, ret = acl.rt.malloc(size, 0)
            self.check(ret, "malloc out")
            self.dev_out[name] = ptr

        self.in_ds = acl.mdl.create_dataset()
        for name in self.in_names:
            size = self.in_sizes[self.in_names.index(name)]
            self.in_ds, ret = acl.mdl.add_dataset_buffer(
                self.in_ds, acl.create_data_buffer(self.dev_in[name], size))
            self.check(ret, "add in buffer")
        self.out_ds = acl.mdl.create_dataset()
        for name in self.out_names:
            size = self.out_sizes[self.out_names.index(name)]
            self.out_ds, ret = acl.mdl.add_dataset_buffer(
                self.out_ds, acl.create_data_buffer(self.dev_out[name], size))
            self.check(ret, "add out buffer")

    def check(self, value, what):
        if isinstance(value, tuple):
            value = value[-1]
        if value != 0:
            raise RuntimeError("%s 失败: ret=%s msg=%s"
                               % (what, value, self.acl.get_recent_err_msg()))

    def run(self, feeds):
        for i, name in enumerate(self.in_names):
            if name in feeds:
                key = name
            else:
                target = norm(name)
                key = None
                for candidate in feeds:
                    if norm(candidate) == target:
                        key = candidate
                        break
                if key is None:
                    raise KeyError("模型输入 %r 没有对应的 feed，现有: %s" % (name, list(feeds)))
            arr = np.ascontiguousarray(feeds[key], dtype=np.float32)
            if arr.nbytes != self.in_sizes[i]:
                raise RuntimeError("输入 %s 大小不符: %d != %d" % (name, arr.nbytes, self.in_sizes[i]))
            self.check(self.acl.rt.memcpy(self.dev_in[name], self.in_sizes[i],
                                          arr.ctypes.data, arr.nbytes, ACL_H2D), "H2D " + name)
        self.check(self.acl.mdl.execute(self.model_id, self.in_ds, self.out_ds), "execute")
        outs = {}
        for i, name in enumerate(self.out_names):
            buf = np.empty(self.out_sizes[i] // 4, dtype=np.float32)
            self.check(self.acl.rt.memcpy(buf.ctypes.data, self.out_sizes[i],
                                          self.dev_out[name], self.out_sizes[i], ACL_D2H),
                       "D2H " + name)
            outs[name] = buf
        return outs


class Rwkv7NpuEngine:
    def __init__(self, model_dir, device=0):
        import acl
        self.acl = acl
        self.model_dir = model_dir
        self.device = device
        self.check(acl.init(), "acl.init")
        self.check(acl.rt.set_device(device), "set_device")
        self.context, ret = acl.rt.create_context(device)
        self.check(ret, "create_context")

        t0 = time.time()
        self.layers = [OmModel(acl, os.path.join(model_dir, "layer%d.om" % i)) for i in range(LAYERS)]
        self.head = OmModel(acl, os.path.join(model_dir, "head.om"))
        print("加载 %d 个层 .om + head 用时 %.1f s" % (LAYERS, time.time() - t0), flush=True)

        self.embedding = self._load_embedding()
        self.states = [(np.zeros((1, HIDDEN), np.float32),
                        np.zeros((1, HIDDEN), np.float32),
                        np.zeros((1, HEADS, HEAD_DIM, HEAD_DIM), np.float32))
                       for _ in range(LAYERS)]

    def check(self, value, what):
        if isinstance(value, tuple):
            value = value[-1]
        if value != 0:
            raise RuntimeError("%s 失败: ret=%s" % (what, value))

    def _load_embedding(self):
        import torch
        from safetensors import safe_open
        index = json.load(open(os.path.join(self.model_dir, "model.safetensors.index.json")))["weight_map"]
        with safe_open(os.path.join(self.model_dir, index["rwkv7.emb.weight"]), framework="pt") as handle:
            tensor = handle.get_tensor("rwkv7.emb.weight")
        # 权重是 bfloat16，numpy 不认识这个类型，先转 float32 再出 numpy
        return tensor.to("cpu").float().numpy()

    def reset_state(self):
        for i in range(LAYERS):
            self.states[i] = (np.zeros((1, HIDDEN), np.float32),
                              np.zeros((1, HIDDEN), np.float32),
                              np.zeros((1, HEADS, HEAD_DIM, HEAD_DIM), np.float32))

    def step(self, token_id):
        x = self.embedding[token_id].reshape(1, 1, HIDDEN)
        v_first = np.zeros((1, 1, HIDDEN), np.float32)
        for i, layer in enumerate(self.layers):
            att_shift, ffn_shift, wkv = self.states[i]
            outs = layer.run({
                "x": x, "v_first": v_first,
                "att_shift_0": att_shift, "ffn_shift_0": ffn_shift, "wkv_0": wkv,
            })
            x = outs[match_name(outs.keys(), "x")].reshape(1, 1, HIDDEN)
            v_first = outs[match_name(outs.keys(), "v_first")].reshape(1, 1, HIDDEN)
            self.states[i] = (
                outs[match_name(outs.keys(), "att_shift_out_0")].reshape(1, HIDDEN),
                outs[match_name(outs.keys(), "ffn_shift_out_0")].reshape(1, HIDDEN),
                outs[match_name(outs.keys(), "wkv_out_0")].reshape(1, HEADS, HEAD_DIM, HEAD_DIM),
            )
        logits = self.head.run({"x": x})[match_name(self.head.out_names, "logits")]
        return logits.reshape(-1).astype(np.float32)


def load_tokenizer(model_dir):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/home/disk/models/rwkv7-2.9b")
    parser.add_argument("--prompt", default="你好")
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--verify", action="store_true", help="与 CPU 参考 logits 比对")
    parser.add_argument("--ref", default="/home/disk/models/rwkv7-2.9b/ref_step.npz")
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.model_dir)
    engine = Rwkv7NpuEngine(args.model_dir)

    if args.verify:
        ref = np.load(args.ref)
        ref_ids = ref["ids"].reshape(-1).tolist()
        ref_logits = ref["logits"]
        print("参考 prompt ids:", ref_ids, flush=True)
        logits = None
        t0 = time.time()
        for tok in ref_ids:
            logits = engine.step(int(tok))
        print("NPU 跑 %d 个 token 用时 %.2f s" % (len(ref_ids), time.time() - t0), flush=True)
        top_npu = np.argsort(-logits)[:5]
        top_ref = np.argsort(-ref_logits)[:5]
        print("%-8s %-12s %-12s" % ("rank", "NPU id/logit", "CPU id/logit"))
        for rank, (a, b) in enumerate(zip(top_npu, top_ref), 1):
            print("%-8d %-12s %-12s" % (rank, "%d/%.4f" % (a, logits[a]), "%d/%.4f" % (b, ref_logits[b])))
        same = list(top_npu) == list(top_ref)
        diff = float(np.max(np.abs(logits - ref_logits)))
        print("top-5 是否一致: %s；logits 最大绝对误差: %.4f" % (same, diff))
        return 0

    ids = tokenizer.encode(args.prompt).ids
    print("prompt=%r -> %d token: %s" % (args.prompt, len(ids), ids), flush=True)
    t0 = time.time()
    logits = None
    for tok in ids:
        logits = engine.step(int(tok))
    print("prefill %d token 用时 %.2f s" % (len(ids), time.time() - t0), flush=True)

    out_ids = []
    t0 = time.time()
    for _ in range(args.tokens):
        nxt = int(np.argmax(logits))
        out_ids.append(nxt)
        logits = engine.step(nxt)
    dt = time.time() - t0
    print("生成 %d token 用时 %.2f s（%.2f token/s）"
          % (len(out_ids), dt, len(out_ids) / dt))
    print("生成文本: %r" % tokenizer.decode(out_ids))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
