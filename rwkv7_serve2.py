#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RWKV-7 2.9B NPU 推理服务 v2：零拷贝串层。

与 v1 的区别：
  v1: 每层 D2H 读回 state，再 H2D 写进下一层 —— 每 token 32 次同步往返 + ~43MB 搬运
  v2: 层间 x/v_first 直接复用上一层的设备输出缓冲；state 在设备上原地更新
      → 每 token 只有 1 次 H2D（embedding 10KB）+ 1 次 D2H（logits 262KB）

用法同 v1：--verify-ref / --bench N / --serve
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
X_BYTES = HIDDEN * 4
WKV_BYTES = HEADS * HEAD_DIM * HEAD_DIM * 4
LOGITS_BYTES = VOCAB * 4


def norm(name):
    return re.sub(r"\.\d+$", "", name.split(":")[-1])


class OmModel:
    """包装一个 .om，输入/输出缓冲由外部（Engine）提供，便于做零拷贝串联。"""

    def __init__(self, acl, path):
        self.acl = acl
        self.model_id, ret = acl.mdl.load_from_file(path)
        if ret != 0:
            raise RuntimeError("加载 %s 失败: %s" % (path, ret))
        self.desc = acl.mdl.create_desc()
        acl.mdl.get_desc(self.desc, self.model_id)
        n_in = acl.mdl.get_num_inputs(self.desc)
        n_out = acl.mdl.get_num_outputs(self.desc)
        self.in_names = [norm(acl.mdl.get_input_name_by_index(self.desc, i)) for i in range(n_in)]
        self.out_names = [norm(acl.mdl.get_output_name_by_index(self.desc, i)) for i in range(n_out)]
        self.in_sizes = [acl.mdl.get_input_size_by_index(self.desc, i) for i in range(n_in)]
        self.out_sizes = [acl.mdl.get_output_size_by_index(self.desc, i) for i in range(n_out)]
        self.in_dev = [0] * n_in
        self.out_dev = [0] * n_out
        self.in_ds = None
        self.out_ds = None

    def attach(self, in_ptrs, out_ptrs):
        acl = self.acl
        self.in_dev = in_ptrs
        self.out_dev = out_ptrs
        self.in_ds = acl.mdl.create_dataset()
        for ptr, size in zip(in_ptrs, self.in_sizes):
            self.in_ds, _ = acl.mdl.add_dataset_buffer(self.in_ds, acl.create_data_buffer(ptr, size))
        self.out_ds = acl.mdl.create_dataset()
        for ptr, size in zip(out_ptrs, self.out_sizes):
            self.out_ds, _ = acl.mdl.add_dataset_buffer(self.out_ds, acl.create_data_buffer(ptr, size))

    def execute(self):
        ret = self.acl.mdl.execute(self.model_id, self.in_ds, self.out_ds)
        if ret not in (0, (0,)):
            raise RuntimeError("execute 失败: %s" % (ret,))


class Engine:
    def __init__(self, model_dir, device=0, suffix="", head_suffix=None, plan=None):
        """suffix='_q' 时优先加载 layer{i}_q.om / head_q.om（int8），
        缺哪个就自动回退到对应的 fp16 模型 —— 量化构建到一半也能直接跑。

        plan（可选）：逐层选择用哪套模型，形如
            {"suffix": "_cp_q", "layers": [0, 1, 7], "head": true}
        未列入 layers 的层、以及 head=false 时使用 fp16。用于混合精度实验：
        每层 fp16 与 int8 版本都在磁盘上，因此换计划不需要重新编译。
        """
        import acl
        self.acl = acl
        self.model_dir = model_dir
        self.suffix = suffix
        self.head_suffix = suffix if head_suffix is None else head_suffix
        self.plan_layers = None
        self.plan_head = None
        if plan:
            self.suffix = plan.get("suffix", suffix) or suffix
            self.plan_layers = set(int(x) for x in plan.get("layers", []))
            self.plan_head = bool(plan.get("head", False))
            self.head_suffix = self.suffix
        self.n_quant = 0
        self.n_plain = 0
        t0 = time.time()
        acl.init()
        acl.rt.set_device(device)
        self.context, ret = acl.rt.create_context(device)
        if ret != 0:
            raise RuntimeError("create_context 失败")

        # ---- 设备缓冲 ----
        self.x_in_dev = acl.rt.malloc(X_BYTES, 0)[0]        # 喂入 embedding
        self.vf_zero_dev = acl.rt.malloc(X_BYTES, 0)[0]     # v_first 初值（全 0）
        self.layer_x = [acl.rt.malloc(X_BYTES, 0)[0] for _ in range(LAYERS)]
        self.layer_vf = [acl.rt.malloc(X_BYTES, 0)[0] for _ in range(LAYERS)]
        self.att = [acl.rt.malloc(X_BYTES, 0)[0] for _ in range(LAYERS)]
        self.ffn = [acl.rt.malloc(X_BYTES, 0)[0] for _ in range(LAYERS)]
        self.wkv = [acl.rt.malloc(WKV_BYTES, 0)[0] for _ in range(LAYERS)]
        self.logits_dev = acl.rt.malloc(LOGITS_BYTES, 0)[0]

        zero_x = np.zeros(HIDDEN, np.float32)
        zero_w = np.zeros(HEADS * HEAD_DIM * HEAD_DIM, np.float32)
        for i in range(LAYERS):
            acl.rt.memcpy(self.att[i], X_BYTES, zero_x.ctypes.data, X_BYTES, ACL_H2D)
            acl.rt.memcpy(self.ffn[i], X_BYTES, zero_x.ctypes.data, X_BYTES, ACL_H2D)
            acl.rt.memcpy(self.wkv[i], WKV_BYTES, zero_w.ctypes.data, WKV_BYTES, ACL_H2D)
        acl.rt.memcpy(self.vf_zero_dev, X_BYTES, zero_x.ctypes.data, X_BYTES, ACL_H2D)

        # ---- 逐层绑定 ----
        self.layers = []
        for i in range(LAYERS):
            if self.plan_layers is not None:
                layer_suffix = self.suffix if i in self.plan_layers else ""
            else:
                layer_suffix = self.suffix
            om = OmModel(acl, self._om_path("layer%d" % i, layer_suffix))
            src_x = self.x_in_dev if i == 0 else self.layer_x[i - 1]
            src_vf = self.vf_zero_dev if i == 0 else self.layer_vf[i - 1]
            slot = {
                "x": src_x,
                "v_first": src_vf,
                "att_shift_0": self.att[i],
                "ffn_shift_0": self.ffn[i],
                "wkv_0": self.wkv[i],
            }
            dst = {
                "x": self.layer_x[i],
                "v_first": self.layer_vf[i],
                "att_shift_out_0": self.att[i],   # 设备上原地更新
                "ffn_shift_out_0": self.ffn[i],
                "wkv_out_0": self.wkv[i],
            }
            om.attach([slot[n] for n in om.in_names], [dst[n] for n in om.out_names])
            self.layers.append(om)

        if self.plan_head is not None:
            head_suffix = self.head_suffix if self.plan_head else ""
        else:
            head_suffix = self.head_suffix
        self.head = OmModel(acl, self._om_path("head", head_suffix))
        self.head.attach([self.layer_x[LAYERS - 1]], [self.logits_dev])

        self.embedding = self._load_embedding(model_dir)
        self.x_host = np.zeros(HIDDEN, np.float32)
        self.logits = np.empty(VOCAB, np.float32)
        print("加载 32 层 + head 用时 %.1f s（零拷贝绑定；int8 %d 层，fp16 %d 层）"
              % (time.time() - t0, self.n_quant, self.n_plain), flush=True)

    def _om_path(self, base, suffix=None):
        """有 int8 版就用 int8，否则回退 fp16。"""
        want = self.suffix if suffix is None else suffix
        if want:
            cand = os.path.join(self.model_dir, base + want + ".om")
            if os.path.exists(cand):
                self.n_quant += 1
                return cand
        self.n_plain += 1
        return os.path.join(self.model_dir, base + ".om")

    def _load_embedding(self, model_dir):
        import torch
        from safetensors import safe_open
        index = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
        with safe_open(os.path.join(model_dir, index["rwkv7.emb.weight"]), framework="pt") as handle:
            tensor = handle.get_tensor("rwkv7.emb.weight")
        return tensor.to("cpu").float().numpy()

    def reset_state(self):
        acl = self.acl
        zero_x = np.zeros(HIDDEN, np.float32)
        zero_w = np.zeros(HEADS * HEAD_DIM * HEAD_DIM, np.float32)
        for i in range(LAYERS):
            acl.rt.memcpy(self.att[i], X_BYTES, zero_x.ctypes.data, X_BYTES, ACL_H2D)
            acl.rt.memcpy(self.ffn[i], X_BYTES, zero_x.ctypes.data, X_BYTES, ACL_H2D)
            acl.rt.memcpy(self.wkv[i], WKV_BYTES, zero_w.ctypes.data, WKV_BYTES, ACL_H2D)

    # ---- 状态快照：把系统提示词处理完的状态存下来，之后每轮直接从这里起步 ----
    def export_state(self):
        acl = self.acl
        snap = {}
        for i in range(LAYERS):
            for name, ptr, size in (("att", self.att[i], X_BYTES),
                                    ("ffn", self.ffn[i], X_BYTES),
                                    ("wkv", self.wkv[i], WKV_BYTES)):
                buf = np.empty(size // 4, np.float32)
                acl.rt.memcpy(buf.ctypes.data, size, ptr, size, ACL_D2H)
                snap["%s_%d" % (name, i)] = buf
        return snap

    def import_state(self, snap):
        acl = self.acl
        for i in range(LAYERS):
            for name, ptr, size in (("att", self.att[i], X_BYTES),
                                    ("ffn", self.ffn[i], X_BYTES),
                                    ("wkv", self.wkv[i], WKV_BYTES)):
                arr = np.ascontiguousarray(snap["%s_%d" % (name, i)], np.float32)
                acl.rt.memcpy(ptr, size, arr.ctypes.data, size, ACL_H2D)

    def step(self, token_id):
        acl = self.acl
        np.copyto(self.x_host, self.embedding[token_id])
        acl.rt.memcpy(self.x_in_dev, X_BYTES, self.x_host.ctypes.data, X_BYTES, ACL_H2D)
        for om in self.layers:
            om.execute()
        self.head.execute()
        acl.rt.memcpy(self.logits.ctypes.data, LOGITS_BYTES, self.logits_dev, LOGITS_BYTES, ACL_D2H)
        return self.logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/home/disk/models/rwkv7-2.9b")
    parser.add_argument("--verify-ref", action="store_true")
    parser.add_argument("--ref", default="/home/disk/models/rwkv7-2.9b/ref_step.npz")
    parser.add_argument("--bench", type=int, default=0)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--suffix", default="", help="'_q' 表示用 int8 模型（缺层自动回退）")
    parser.add_argument("--head-suffix", default=None, help="输出头单独指定后缀")
    parser.add_argument("--plan", default=None,
                        help='逐层计划 JSON，如 \'{"suffix":"_cp_q","layers":[0,1],"head":false}\'')
    args = parser.parse_args()

    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(os.path.join(args.model_dir, "tokenizer.json"))
    plan = json.loads(args.plan) if args.plan else None
    engine = Engine(args.model_dir, suffix=args.suffix, head_suffix=args.head_suffix, plan=plan)

    def run(ids, n, stream=False):
        engine.reset_state()
        logits = None
        out_ids = []
        rng = np.random.default_rng(0)
        for tok in ids:
            logits = engine.step(int(tok))
        for _ in range(n):
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
        _, logits = run(ids, 0)
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
        run(ids, 2)
        t0 = time.time()
        out_ids, _ = run(ids, args.bench)
        dt = time.time() - t0
        print("生成 %d token 用时 %.3f s → %.2f token/s（%.1f ms/token）"
              % (args.bench, dt, args.bench / dt, dt / args.bench * 1000))
        print("文本: %r" % tokenizer.decode(out_ids))
        return 0

    if args.serve:
        print("READY", flush=True)
        for line in sys.stdin:
            line = line.strip()
            if line == "/quit":
                break
            if not line:
                continue
            n = args.tokens
            if "#" in line:
                line, _, tail = line.rpartition("#")
                try:
                    n = int(tail)
                except ValueError:
                    pass
            ids = tokenizer.encode(line).ids
            t0 = time.time()
            run(ids, n, stream=True)
            dt = time.time() - t0
            print("\n[%d token, %.2f s, %.2f token/s, prompt %d token]"
                  % (n, dt, n / dt if dt else 0, len(ids)), flush=True)
        return 0

    print("请指定 --verify-ref / --bench N / --serve")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
