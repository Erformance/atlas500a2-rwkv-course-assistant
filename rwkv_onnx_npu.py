#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RWKV-4/5 单步 ONNX 模型在 Atlas 500 A2 上的推理脚本。

两种后端：
  acl : ATC 编译出的 .om，用 pyACL 在 310B NPU 上执行
  ort : 原始 .onnx，用 onnxruntime 在 CPU 上执行（做数值对照）

用法示例：
  # NPU 上跑
  python rwkv_onnx_npu.py --backend acl --model rwkv04.om --prompt "User: hello" --tokens 32
  # CPU 上跑同一模型的 ONNX，用于对照
  python rwkv_onnx_npu.py --backend ort --model model_fixed.onnx --prompt "User: hello" --tokens 32
"""

import argparse
import importlib.util
import sys
import time

import numpy as np

STATE_INPUTS = ["xx_att", "aa_att", "bb_att", "pp_att", "xx_ffn"]


def load_tokenizer(tokenizer_py, vocab_path):
    spec = importlib.util.spec_from_file_location("rwkv_tokenizer", tokenizer_py)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TRIE_TOKENIZER(vocab_path)


class OrtBackend:
    name = "ort(CPU)"

    def __init__(self, model_path, device=None):
        import onnxruntime as ort
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.in_names = [i.name for i in self.session.get_inputs()]
        self.out_names = [o.name for o in self.session.get_outputs()]
        print("模型输入:", self.in_names)
        print("模型输出:", self.out_names)

    def run(self, feeds):
        outs = self.session.run(self.out_names, feeds)
        return dict(zip(self.out_names, outs))

    def close(self):
        pass


class AclBackend:
    name = "acl(NPU)"
    ACL_DT_FLOAT = 0
    ACL_DT_INT32 = 3

    def __init__(self, model_path, device=0):
        import acl
        self.acl = acl
        self.device = device
        self.ret(self.acl.init(), "acl.init")
        self.ret(self.acl.rt.set_device(device), "set_device")
        self.context, ret = self.acl.rt.create_context(device)
        self.ret(ret, "create_context")
        self.model_id, ret = self.acl.mdl.load_from_file(model_path)
        self.ret(ret, "load_from_file")
        self.desc = self.acl.mdl.create_desc()
        self.ret(self.acl.mdl.get_desc(self.desc, self.model_id), "get_desc")

        n_in = self.acl.mdl.get_num_inputs(self.desc)
        n_out = self.acl.mdl.get_num_outputs(self.desc)
        self.in_names = [self.acl.mdl.get_input_name_by_index(self.desc, i) for i in range(n_in)]
        self.out_names = [self.acl.mdl.get_output_name_by_index(self.desc, i) for i in range(n_out)]
        self.in_sizes = [self.acl.mdl.get_input_size_by_index(self.desc, i) for i in range(n_in)]
        self.out_sizes = [self.acl.mdl.get_output_size_by_index(self.desc, i) for i in range(n_out)]
        self.in_dtypes = [self.acl.mdl.get_input_data_type(self.desc, i) for i in range(n_in)]
        self.out_dtypes = [self.acl.mdl.get_output_data_type(self.desc, i) for i in range(n_out)]
        print("模型输入:", list(zip(self.in_names, self.in_sizes, self.in_dtypes)))
        print("模型输出:", list(zip(self.out_names, self.out_sizes, self.out_dtypes)))

        self.dev_in = {}
        for name, size in zip(self.in_names, self.in_sizes):
            ptr, ret = self.acl.rt.malloc(size, 0)
            self.ret(ret, "malloc in " + name)
            self.dev_in[name] = ptr
        self.dev_out = {}
        for name, size in zip(self.out_names, self.out_sizes):
            ptr, ret = self.acl.rt.malloc(size, 0)
            self.ret(ret, "malloc out " + name)
            self.dev_out[name] = ptr

        self.in_ds = self.acl.mdl.create_dataset()
        for name in self.in_names:
            size = self.in_sizes[self.in_names.index(name)]
            buf = self.acl.create_data_buffer(self.dev_in[name], size)
            self.in_ds, ret = self.acl.mdl.add_dataset_buffer(self.in_ds, buf)
            self.ret(ret, "add in buffer " + name)
        self.out_ds = self.acl.mdl.create_dataset()
        for name in self.out_names:
            size = self.out_sizes[self.out_names.index(name)]
            buf = self.acl.create_data_buffer(self.dev_out[name], size)
            self.out_ds, ret = self.acl.mdl.add_dataset_buffer(self.out_ds, buf)
            self.ret(ret, "add out buffer " + name)

    def ret(self, value, what):
        if isinstance(value, tuple):
            value = value[-1]
        if value != 0:
            raise RuntimeError("%s 失败: ret=%s msg=%s" % (what, value, self.acl.get_recent_err_msg()))

    def run(self, feeds):
        for i, name in enumerate(self.in_names):
            arr = np.ascontiguousarray(feeds[name])
            if arr.nbytes != self.in_sizes[i]:
                raise RuntimeError("输入 %s 大小不符: %d != %d" % (name, arr.nbytes, self.in_sizes[i]))
            self.ret(
                self.acl.rt.memcpy(
                    self.dev_in[name], self.in_sizes[i], arr.ctypes.data, arr.nbytes, 1
                ),
                "H2D " + name,
            )
        self.ret(self.acl.mdl.execute(self.model_id, self.in_ds, self.out_ds), "execute")
        outs = {}
        for i, name in enumerate(self.out_names):
            buf = np.empty(self.out_sizes[i] // 4, dtype=np.float32)
            self.ret(
                self.acl.rt.memcpy(
                    buf.ctypes.data, self.out_sizes[i], self.dev_out[name], self.out_sizes[i], 2
                ),
                "D2H " + name,
            )
            outs[name] = buf
        return outs

    def close(self):
        for ptr in list(self.dev_in.values()) + list(self.dev_out.values()):
            self.acl.rt.free(ptr)
        self.acl.mdl.destroy_dataset(self.in_ds)
        self.acl.mdl.destroy_dataset(self.out_ds)
        self.acl.mdl.unload(self.model_id)
        self.acl.mdl.destroy_desc(self.desc)
        self.acl.rt.destroy_context(self.context)
        self.acl.rt.reset_device(self.device)
        self.acl.finalize()


def pick(outs, key):
    """按名字取输出；ATC 编译后名字会带前缀（如 /Concat:0:xx_att_r），所以支持后缀匹配。"""
    if key in outs:
        return outs[key]
    for name, value in outs.items():
        if name.endswith(key):
            return value
    for name, value in outs.items():
        if key in name:
            return value
    raise KeyError("找不到输出 %r，现有输出: %s" % (key, list(outs)))


def build_state(outs):
    return {name: pick(outs, name + "_r").reshape(24, 1024).astype(np.float32) for name in STATE_INPUTS}


def logits_of(outs, vocab_size=65536):
    """词表 logits 是唯一大小为 vocab_size 的输出（state 输出都是 24*1024）。"""
    for value in outs.values():
        if value.size == vocab_size:
            return value.astype(np.float32)
    raise KeyError("找不到 %d 维的 logits 输出" % vocab_size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["acl", "ort"], required=True)
    parser.add_argument("--model", required=True, help=".om（acl）或 .onnx（ort）")
    parser.add_argument("--prompt", default="User: hello\n\nAssistant:")
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--vocab", default="/home/disk/models/rwkv-onnx-0.4b/rwkv_vocab_v20230424.txt")
    parser.add_argument("--tokenizer-py", default="/home/disk/models/rwkv-onnx-0.4b/rwkv_tokenizer.py")
    parser.add_argument("--dump-ids", help="把生成的 token id 写到文件")
    parser.add_argument("--temperature", type=float, default=0.0, help="0 表示贪心")
    parser.add_argument("--top-k", type=int, default=0)
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.tokenizer_py, args.vocab)
    backend = AclBackend(args.model, args.device) if args.backend == "acl" else OrtBackend(args.model)

    try:
        ids = tokenizer.encode(args.prompt)
        print("prompt=%r -> %d 个 token: %s" % (args.prompt, len(ids), ids))

        state = {name: np.zeros((24, 1024), dtype=np.float32) for name in STATE_INPUTS}
        outs = None
        t0 = time.time()
        for tok in ids:
            outs = backend.run({"idx": np.array([tok], dtype=np.int32), **state})
            state = build_state(outs)
        prefill_s = time.time() - t0
        print("prefill %d token 用时 %.3f s" % (len(ids), prefill_s))

        logits = logits_of(outs)
        out_ids = []
        pieces = []
        t0 = time.time()
        rng = np.random.default_rng(0)
        for step in range(args.tokens):
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
            piece = tokenizer.decode([nxt])
            pieces.append(piece)
            outs = backend.run({"idx": np.array([nxt], dtype=np.int32), **state})
            state = build_state(outs)
            logits = logits_of(outs)
            print("  step %2d: id=%d %r top1_logit=%.4f" % (step, nxt, piece, float(logits[nxt])))
        gen_s = time.time() - t0
        text = tokenizer.decode(out_ids)
        print("-" * 60)
        print("生成文本: %r" % text)
        print("生成 %d token 用时 %.3f s（%.2f token/s）" % (len(out_ids), gen_s, len(out_ids) / gen_s))
        print("完整文本: %r" % (args.prompt + text))
        if args.dump_ids:
            with open(args.dump_ids, "w") as fh:
                fh.write(",".join(str(i) for i in out_ids))
            print("token id 已写入 %s" % args.dump_ids)
    finally:
        backend.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
