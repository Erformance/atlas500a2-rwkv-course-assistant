#!/bin/bash
# 尾队列：等前一个串行队列（queue_next.sh）结束，再跑 P3-E（Gramian 一致性/耗时）。
# 用法：setsid nohup bash queue_tail.sh < /dev/null > /dev/null 2>&1 &

MODEL_DIR=/home/disk/models/rwkv7-2.9b
LOG=$MODEL_DIR/queue_tail.log

cd "$MODEL_DIR" || exit 1
exec >> "$LOG" 2>&1

echo
echo "########## 尾队列开始 $(date '+%F %T') ##########"
# 9-28 教训：只检查 queue_next.sh 不够——它退出后子链（p4_item_cpu.sh /
# p4_arc_hella_chain.sh / p3d_chain.sh）可能还在跑，尾链会误判"前序已结束"而并发。
# 现在把子链一并等干净，并且以 experiment.lock 为准。
busy() {
  pgrep -f queue_next.sh > /dev/null && return 0
  pgrep -f p4_item_cpu.sh > /dev/null && return 0
  pgrep -f p4_arc_hella_chain.sh > /dev/null && return 0
  pgrep -f p3d_chain.sh > /dev/null && return 0
  [ -f "$MODEL_DIR/experiment.lock" ] && return 0
  return 1
}

for i in $(seq 1 720); do
  busy || break
  sleep 30
done
if busy; then
  echo "[$(date '+%T')] 前序仍忙（含锁），尾队列退出，不做并发"
  exit 1
fi
echo "[$(date '+%T')] 前序队列已结束，开始 P3-E"

bash p3e_chain.sh
echo "########## 尾队列结束 $(date '+%F %T') ##########"
