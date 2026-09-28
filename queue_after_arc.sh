#!/bin/bash
# 等 P4 ⑤ 均衡档评测链跑完（并且 experiment.lock 已释放），再补跑 P3-E。
# 9-28 的教训：任何"等一会儿就硬上"的做法都会造成并发，所以这里两个条件都等，
# 等满 8 小时仍未满足就退出。
# 日志：queue_after_arc.log
#
# 用法：setsid nohup bash queue_after_arc.sh < /dev/null > /dev/null 2>&1 &

MODEL_DIR=/home/disk/models/rwkv7-2.9b
LOG=$MODEL_DIR/queue_after_arc.log

cd "$MODEL_DIR" || exit 1
exec >> "$LOG" 2>&1

echo
echo "########## P3-E 补跑等待器启动 $(date '+%F %T') ##########"

busy() {
  pgrep -f p4_arc_hella_chain.sh > /dev/null && return 0
  pgrep -f p4_score_server.py > /dev/null && return 0
  pgrep -f finish_queue.sh > /dev/null && return 0
  [ -f "$MODEL_DIR/experiment.lock" ] && return 0
  return 1
}

for i in $(seq 1 480); do
  busy || break
  echo "  均衡档评测仍在跑，等待（$i）"
  sleep 60
done
if busy; then
  echo "等了 8 小时仍未空，退出（不做并发）"
  exit 1
fi

echo "[$(date '+%T')] 均衡档已结束，开始补跑 P3-E"
bash p3e_chain.sh
echo "########## P3-E 补跑结束 $(date '+%F %T') ##########"
ls -l p3e_gramian.json 2>&1 | tail -2
