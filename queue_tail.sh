#!/bin/bash
# 尾队列：等前一个串行队列（queue_next.sh）结束，再跑 P3-E（Gramian 一致性/耗时）。
# 用法：setsid nohup bash queue_tail.sh < /dev/null > /dev/null 2>&1 &

MODEL_DIR=/home/disk/models/rwkv7-2.9b
LOG=$MODEL_DIR/queue_tail.log

cd "$MODEL_DIR" || exit 1
exec >> "$LOG" 2>&1

echo
echo "########## 尾队列开始 $(date '+%F %T') ##########"
for i in $(seq 1 720); do
  pgrep -f queue_next.sh > /dev/null || break
  sleep 30
done
echo "[$(date '+%T')] 前序队列已结束，开始 P3-E"

bash p3e_chain.sh
echo "########## 尾队列结束 $(date '+%F %T') ##########"
