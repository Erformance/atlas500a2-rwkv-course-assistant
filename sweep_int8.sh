#!/bin/bash
# int8 层数梯度扫描：找出"还能保持回答质量"的最大 int8 层数。
# 结果写进 sweep_int8.log
cd /home/disk/models/rwkv7-2.9b || exit 1
source /home/disk/cann80base/ascend-toolkit/set_env.sh
PY=/home/disk/miniconda3/envs/npu22/bin/python

for keep in 0 4 8 12 16 24 32; do
  bash toggle_int8.sh restore > /dev/null
  bash toggle_int8.sh "$keep" > /dev/null
  echo "########## int8 层数 = $keep   $(date +%H:%M:%S)"
  echo "--- 问题1"
  "$PY" rwkv7_chat.py --ask "什么是人工智能？" --tokens 96 --suffix _q
  echo "--- 问题2"
  "$PY" rwkv7_chat.py --ask "请用三句话解释牛顿第二定律。" --tokens 96 --suffix _q
done

bash toggle_int8.sh restore
echo "########## 扫描完成 $(date +%H:%M:%S)"
