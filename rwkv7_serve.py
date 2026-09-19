#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RWKV-7 G1j 2.9B 在 Atlas 500 A2（310B）上的常驻推理服务。

相对第一版的优化：
  * 加载时预计算所有 输入/输出 → 缓冲区 的映射，推理循环里零名字匹配、零字典构造
  * 所有 state 缓冲区预分配复用，循环内不产生新对象
  * 常驻进程：模型只加载一次（省掉每次 25~35s 的冷启动）

用法：
  python rwkv7_serve.py --verify-ref          # 与 CPU 参考 logits 比对
  python rwkv7_serve.py --bench 5             # 跑 5 token 测速
  python rwkv7_serve.py --serve               # 常驻：从 stdin 读 prompt，流式输出
"""

import argparse
import json
import os
import re
import sys
import time

import numpy as np

HIDDEN, HEADS, HEAD_DIM, LAYERS, VOCAB = 2560, 40, 64, 32, 65536
ACL_H2D, ACL_D2H = 1, 2

# 语义槽位
SLOT_X, SLOT_VF = 0, 1
SLOT_ATT = 2
SLOT_FFN = SLOT_ATT + LAYERS
SLOT_WKV = SLOT_FFN + LAYERS
NSLOT = SLOT_WKV + LAYERS


def norm(name):
    return re.sub(r"\.\d+$", "", name.split(":")[-1])


class OmModel:
    """一个 .om + 预计算好的 IO 映射。"""

    def __init__(self, acl, path):
        self.acl = acl
        self.model_id, ret = acl.mdl.load_from_file(path)
        if ret != 0:
            raise RuntimeError("加载 %s 失败: %s" % (path, ret))
        self.desc = acl.mdl.create_desc()
        acl.mdl.get_desc(self.desc, self.model_id)

        n_in = acl.mdl.get_num_inputs(self.desc)
        n_out = acl.mdl.get_num_outputs(self.desc)
        in_names = [acl.mdl.get_input_name_by_index(self.desc, i) for i in range(n_in)]
        out_names = [acl.mdl.get_output_name_by_index(self.desc, i) for i in range(n_out)]
        in_sizes = [acl.mdl.get_input_size_by_index(self.desc, i) for i in range(n_in)]
        out_sizes = [acl.mdl.get_output_size_by_index(self.desc, i) for i in range(n_out)]

        self.in_dev = [acl.rt.malloc(s, 0)[0] for s in in_sizes]
        self.out_dev = [acl.rt.malloc(s, 0)[0] for s in out_sizes]
        self.in_ds = acl.mdl.create_dataset()
        for ptr, size in zip(self.in_dev, in_sizes):
            self.in_ds, _ = acl.mdl.add_dataset_buffer(self.in_ds, acl.create_data_buffer(ptr, size))
        self.out_ds = acl.mdl.create_dataset()
        for ptr, size in zip(self.out_dev, out_sizes):
            self.out_ds, _ = acl.mdl.add_dataset_buffer(self.out_ds, acl.create_data_buffer(ptr, size))

        self.in_names = in_names
        self.in_sizes = in_sizes
        self.out_names = out_names
        self.out_sizes = out_sizes
        self.in_slot = [None] * n_in      # 输入序号 → 语义槽位
        self.out_slot = [None] * n_out    # 输出序号 → 语义槽位

    def bind(self, slot_of):
        """根据名字把输入/输出绑定到语义槽位。slot_of(name)->slot or None"""
        for i, name in enumerate(self.in_names):
            self.in_slot[i] = slot_of(norm(name))
        for i, name in enumerate(self.out_names):
            self.out_slot[i] = slot_of(norm(name))


class Engine:
    def __init__(self, model_dir, device=0):
        import acl
        self.acl = acl
        t0 = time.time()
        acl.init()
        acl.rt.set_device(device)
        self.context, ret = acl.rt.create_context(device)
        if ret != 0:
            raise RuntimeError("create_context 失败")

        self.layers = []
        for i in range(LAYERS):
            om = OmModel(acl, os.path.join(model_dir, "layer%d.om" % i))
            om.bind(lambda n, i=i: self._layer_slot(n, i))
            self.layers.append(om)
        self.head = OmModel(acl, os.path.join(model_dir, "head.om"))
        self.head.bind(lambda n: SLOT_X if n == "x" else None)

        # 预分配所有语义槽位的 host 缓冲
        self.slots = []
        for _ in range(NSLOT):
            self.slots.append(None)
        self.slots[SLOT_X] = np.zeros((1, 1, HIDDEN), np.float32)
        self.slots[SLOT_VF] = np.zeros((1, 1, HIDDEN), np.float32)
        for i in range(LAYERS):
            self.slots[SLOT_ATT + i] = np.zeros((1, HIDDEN), np.float32)
            self.slots[SLOT_FFN + i] = np.zeros((1, HIDDEN), np.float32)
            self.slots[SLOT_WKV + i] = np.zeros((1, HEADS, HEAD_DIM, HEAD_DIM), np.float32)
        self.slot_ptrs = [None if s is None else s.ctypes.data for s in self.slots]

        self.embedding = self._load_embedding(model_dir)
        self.logits = np.empty(VOCAB, np.float32)
        print("加载 32 层 + head 用时 %.1f s" % (time.time() - t0), flush=True)

    @staticmethod
    def _layer_slot(name, layer):
        if name == "x":
            return SLOT_X
        if name == "v_first":
            return SLOT_VF
        if name == "att_shift_0" or name == "att_shift_out_0":
            return SLOT_ATT + layer
        if name == "ffn_shift_0" or name == "ffn_shift_out_0":
            return SLOT_FFN + layer
        if name == "wkv_0" or name == "wkv_out_0":
            return SLOT_WKV + layer
        return None

    def _load_embedding(self, model_dir):
        import torch
        from safetensors import safe_open
        index = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
        with safe_open(os.path.join(model_dir, index["rwkv7.emb.weight"]), framework="pt") as handle:
            tensor = handle.get_tensor("rwkv7.emb.weight")
        return tensor.to("cpu").float().numpy()

    def reset_state(self):
        self.slots[SLOT_X][...] = 0
        self.slots[SLOT_VF][...] = 0
        for i in range(LAYERS):
            self.slots[SLOT_ATT + i][...] = 0
            self.slots[SLOT_FFN + i][...] = 0
            self.slots[SLOT_WKV + i][...] = 0

    def step(self, token_id):
        acl = self.acl
        self.slots[SLOT_X][0, 0] = self.embedding[token_id]
        self.slots[SLOT_VF][...] = 0
        for om in self.layers:
            for i, slot in enumerate(om.in_slot):
                if slot is None:
                    continue
                size = om.in_sizes[i]
                acl.rt.memcpy(om.in_dev[i], size, self.slot_ptrs[slot], size, ACL_H2D)
            acl.mdl.execute(om.model_id, om.in_ds, om.out_ds)
            for i, slot in enumerate(om.out_slot):
                if slot is None:
                    continue
                size = om.out_sizes[i]
                acl.rt.memcpy(self.slot_ptrs[slot], size, om.out_dev[i], size, ACL_D2H)
        om = self.head
        size = om.in_sizes[0]
        acl.rt.memcpy(om.in_dev[0], size, self.slot_ptrs[SLOT_X], size, ACL_H2D)
        acl.mdl.execute(om.model_id, om.in_ds, om.out_ds)
        acl.rt.memcpy(self.logits.ctypes.data, VOCAB * 4, om.out_dev[0], VOCAB * 4, ACL_D2H)
        return self.logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/home/disk/models/rwkv7-2.9b")
    parser.add_argument("--verify-ref", action="store_true")
    parser.add_argument("--ref", default="/home/disk/models/rwkv7-2.9b/ref_step.npz")
    parser.add_argument("--bench", type=int, default=0, help="生成 N 个 token 测速")
    parser.add_argument("--serve", action="store_true", help="常驻服务：stdin 逐行读 prompt")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    args = parser.parse_args()

    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(os.path.join(args.model_dir, "tokenizer.json"))
    engine = Engine(args.model_dir)

    def run_tokens(ids, n, stream=False):
        engine.reset_state()
        logits = None
        out_ids = []
        rng = np.random.default_rng(0)
        for i, tok in enumerate(ids):
            logits = engine.step(int(tok))
        for step in range(n):
            if args.temperature <= 0:
                nxt = int(np.argmax(logits))
            else:
                scaled = logits / args.temperature
                if args.top_k > 0:
                    kth = np.sort(scaled)[-args.top_k]
                    scaled = np.where(scaled < kth, -1e30, scaled)
                probs = np.exp(scaled - scaled.max())
                probs /= probs.sum()
                nxt = int(rng.choice(len(probs), p=probs))
            out_ids.append(nxt)
            if stream:
                sys.stdout.write(tokenizer.decode([nxt]))
                sys.stdout.flush()
            logits = engine.step(nxt)
        return out_ids, logits

    if args.verify_ref:
        ref = np.load(args.ref)
        ids = ref["ids"].reshape(-1).tolist()
        t0 = time.time()
        _, logits = run_tokens(ids, 0)
        dt = time.time() - t0
        ref_logits = ref["logits"]
        top_npu = list(np.argsort(-logits)[:5])
        top_ref = list(np.argsort(-ref_logits)[:5])
        print("NPU 跑 %d token 用时 %.3f s（%.1f ms/token）" % (len(ids), dt, dt / len(ids) * 1000))
        print("NPU top5:", [(int(i), round(float(logits[i]), 4)) for i in top_npu])
        print("CPU top5:", [(int(i), round(float(ref_logits[i]), 4)) for i in top_ref])
        print("top-5 一致:", top_npu == top_ref,
              "；最大误差: %.4f" % float(np.max(np.abs(logits - ref_logits))))
        return 0

    if args.bench:
        ids = tokenizer.encode("你好").ids
        t0 = time.time()
        run_tokens(ids, 2)          # 预热
        t1 = time.time()
        out_ids, _ = run_tokens(ids, args.bench)
        dt = time.time() - t1
        print("生成 %d token 用时 %.3f s → %.2f token/s（%.1f ms/token）"
              % (args.bench, dt, args.bench / dt, dt / args.bench * 1000))
        print("文本: %r" % tokenizer.decode(out_ids))
        return 0

    if args.serve:
        print("READY", flush=True)
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            if line == "/quit":
                break
            n = args.tokens
            if "#" in line:
                line, _, tail = line.rpartition("#")
                try:
                    n = int(tail)
                except ValueError:
                    pass
            ids = tokenizer.encode(line).ids
            t0 = time.time()
            run_tokens(ids, n, stream=True)
            dt = time.time() - t0
            print("\n[%d token, %.2f s, %.2f token/s, prompt %d token]"
                  % (n, dt, n / dt if dt else 0, len(ids)), flush=True)
        return 0

    print("请指定 --verify-ref / --bench N / --serve")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
