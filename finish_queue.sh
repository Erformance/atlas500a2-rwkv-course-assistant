#!/bin/bash
# 收尾队列（9-28 重跑版）：严格串行、**跑完不关机**。
#   步骤 1：P3-E（稠密 Gramian vs 结构化递推的一致性 + 离线耗时，纯 CPU，约 20 分钟）
#   步骤 2：P4 ⑤ 均衡档 ARC-Easy / HellaSwag 子集（需要 NPU，约 3.6 小时）
#
# 与昨晚的区别：
#   * 每一步都先等 experiment.lock 空出来；等不到就退出，绝不并发
#     （9-28 的故障就是 P3-E 等锁到点硬上，与 ARC 链抢内存把设备压死）；
#   * 不做自动关机（关机流程在这台设备上不可靠，交给人工）。
# 日志：finish_queue.log
#
# 用法：setsid nohup bash finish_queue.sh < /dev/null > /dev/null 2>&1 &

MODEL_DIR=/home/disk/models/rwkv7-2.9b
LOG=$MODEL_DIR/finish_queue.log

cd "$MODEL_DIR" || exit 1
exec >> "$LOG" 2>&1

echo
echo "########## 收尾队列开始 $(date '+%F %T') ##########"

busy() {
  pgrep -f p3e_chain.sh > /dev/null && return 0
  pgrep -f p4_arc_hella_chain.sh > /dev/null && return 0
  pgrep -f p3d_chain.sh > /dev/null && return 0
  [ -f "$MODEL_DIR/experiment.lock" ] && return 0
  return 1
}

for i in $(seq 1 480); do
  busy || break
  echo "  前序仍忙，等待（$i）"
  sleep 60
done
if busy; then
  echo "前序仍忙，收尾队列退出"
  exit 1
fi

echo "[$(date '+%T')] ===== 步骤 1/2：P3-E（Gramian 一致性/耗时）====="
bash p3e_chain.sh
echo "[$(date '+%T')] 步骤 1 结束"

echo "[$(date '+%T')] ===== 步骤 2/2：P4 ⑤ 均衡档 ARC/HellaSwag ====="
TIERS=balanced bash p4_arc_hella_chain.sh
echo "[$(date '+%T')] 步骤 2 结束"

echo "########## 收尾队列全部结束（不关机）$(date '+%F %T') ##########"
echo "产物："
ls -l p3e_gramian.json p4_eval_balanced_arc_hella.json 2>&1 | tail -4
