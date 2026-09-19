#!/bin/bash
# RWKV-7 2.9B 课程助手（Atlas 500 A2 NPU）一键入口
#   单次提问：  ./ask.sh "什么是人工智能？"          （默认最多生成 128 token）
#   指定长度：  ./ask.sh "什么是人工智能？" 300       （最多生成 300 token）
#               ./ask.sh "什么是人工智能？#300"      （同样效果）
#   连续问答：  ./ask.sh          （输入 /quit 退出；每问后面可加 #数字，如 你好#64）
#
# 注意：常驻服务在跑的时候，这里会直接问服务，不会再加载一份模型
#（一份模型占 5.5GB 内存，两份会把这台 11.5GB 的机器拖死）。

MODEL_DIR=/home/disk/models/rwkv7-2.9b
PY=/home/disk/miniconda3/envs/npu22/bin/python

cd "$MODEL_DIR" || exit 1
source /home/disk/cann80base/ascend-toolkit/set_env.sh

if [ -f service.pid ] && kill -0 "$(cat service.pid)" 2>/dev/null; then
  exec "$PY" ask_http.py "$@"
fi

if [ $# -gt 0 ]; then
  q="$1"
  n="${2:-128}"
  exec "$PY" rwkv7_chat.py --ask "$q" --tokens "$n"
else
  echo "（模型加载约 40 秒，之后可连续提问；/quit 退出）"
  exec "$PY" rwkv7_chat.py --serve
fi
