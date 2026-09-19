#!/bin/bash
# RWKV-7 2.9B 课程助手 —— int8 量化版（Atlas 500 A2 NPU）
#   单次提问：  ./ask_q.sh "什么是人工智能？"         （默认 128 token）
#   指定长度：  ./ask_q.sh "什么是人工智能？" 300
#   连续问答：  ./ask_q.sh
# 说明：int8 模型缺哪层就自动回退 fp16，构建到一半也能用。
source /home/disk/cann80base/ascend-toolkit/set_env.sh
cd /home/disk/models/rwkv7-2.9b || exit 1
if [ $# -gt 0 ]; then
  q="$1"
  n="${2:-128}"
  exec /home/disk/miniconda3/envs/npu22/bin/python rwkv7_chat.py --ask "$q" --tokens "$n" --suffix _q
else
  echo "（模型加载约 40 秒，之后可连续提问；/quit 退出）"
  exec /home/disk/miniconda3/envs/npu22/bin/python rwkv7_chat.py --serve --suffix _q
fi
