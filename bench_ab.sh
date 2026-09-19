#!/bin/bash
# fp16 vs int8 对照测试：同一组问题、同一 token 上限，分别跑一遍，存成两个报告。
# 用法: bash bench_ab.sh [每题 token 上限]   默认 128

TOK=${1:-128}
source /home/disk/cann80base/ascend-toolkit/set_env.sh
cd /home/disk/models/rwkv7-2.9b || exit 1
PY=/home/disk/miniconda3/envs/npu22/bin/python

QUESTIONS=(
  "什么是人工智能？"
  "请用三句话解释牛顿第二定律。"
  "资本主义社会基本矛盾的具体表现有哪些？"
  "写一段 Python 代码，判断一个整数是否为素数。"
  "为什么历史文物值得保护？"
)

for mode in fp16 int8; do
  if [ "$mode" = "int8" ]; then suf="_q"; else suf=""; fi
  echo "=== 跑 $mode（$(date +%H:%M:%S)）token 上限 $TOK"
  printf '%s\n' "${QUESTIONS[@]}" \
    | "$PY" rwkv7_chat.py --serve --tokens "$TOK" --suffix "$suf" \
    > "bench_${mode}.txt" 2>&1
done
echo "=== 完成 $(date +%H:%M:%S)，报告：bench_fp16.txt / bench_int8.txt"
