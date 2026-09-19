#!/bin/bash
# 用真实激活值逐层对比 fp16 / int8 的输出误差，找出被量坏掉的层。
cd /home/disk/models/rwkv7-2.9b || exit 1
source /home/disk/cann80base/ascend-toolkit/set_env.sh
PY=/home/disk/miniconda3/envs/npu22/bin/python

for i in $(seq 0 31); do
  n=$(printf "%02d" "$i")
  echo "### layer$i"
  "$PY" cmp_layer_real.py "layer$i.om" "layer${i}_q.om" "calib/layer${n}.npz" 2>&1 | tail -n 7
done
echo "### 完成"
