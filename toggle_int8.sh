#!/bin/bash
# int8 层数梯度测试：把多余层的 int8 模型挪开，让引擎自动回退 fp16。
#   bash toggle_int8.sh 4        只保留 layer0..3 为 int8
#   bash toggle_int8.sh restore  全部恢复
cd /home/disk/models/rwkv7-2.9b || exit 1

if [ "$1" = "restore" ]; then
  for f in layer*_q.om.off head_q.om.off; do
    [ -e "$f" ] && mv "$f" "${f%.off}"
  done
  echo "已恢复: $(ls layer*_q.om 2>/dev/null | wc -l) 层 int8, head $( [ -e head_q.om ] && echo int8 || echo fp16 )"
  exit 0
fi

keep=${1:-32}
for i in $(seq "$keep" 31); do
  [ -e "layer${i}_q.om" ] && mv "layer${i}_q.om" "layer${i}_q.om.off"
done
[ -e head_q.om ] && mv head_q.om head_q.om.off
echo "当前 int8 层: $(ls layer*_q.om 2>/dev/null | wc -l) 层（应为 $keep）"
