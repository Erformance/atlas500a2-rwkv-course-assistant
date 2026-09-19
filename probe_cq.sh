#!/bin/bash
# 校准版（_cq）int8 层数梯度抽检：找出还能保持回答质量的层数上限。
cd /home/disk/models/rwkv7-2.9b || exit 1
source /home/disk/cann80base/ascend-toolkit/set_env.sh
PY=/home/disk/miniconda3/envs/npu22/bin/python

restore() {
  for f in layer*_cq.om.off; do
    [ -e "$f" ] && mv "$f" "${f%.off}"
  done
}

for keep in 4 8 12 16 20 24 32; do
  restore
  for i in $(seq "$keep" 31); do
    [ -e "layer${i}_cq.om" ] && mv "layer${i}_cq.om" "layer${i}_cq.om.off"
  done
  echo "########## 校准 int8 层数 = $keep   $(date +%H:%M:%S)"
  "$PY" rwkv7_chat.py --ask "什么是人工智能？" --tokens 80 --suffix _cq
done
restore
echo "########## 抽检完成 $(date +%H:%M:%S)"
