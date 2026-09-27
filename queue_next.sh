#!/bin/bash
# 串行队列（协议收尾）：等 P3-D 采臂链结束，再依次跑
#   1) P4 ① 逐题 CPU↔NPU 对照的 CPU 侧 + 比对
#   2) P4 ⑤ ARC-Easy / HellaSwag 子集（fp16 与均衡档）
# 每一步各自写自己的日志；本脚本只负责"一次只跑一个重活"。
# 用法：setsid nohup bash queue_next.sh < /dev/null > /dev/null 2>&1 &

MODEL_DIR=/home/disk/models/rwkv7-2.9b
LOG=$MODEL_DIR/queue_next.log

cd "$MODEL_DIR" || exit 1
exec >> "$LOG" 2>&1

echo
echo "########## 串行队列开始 $(date '+%F %T') ##########"

echo "[$(date '+%T')] 等 P3-D 采臂链结束…"
for i in $(seq 1 360); do
  pgrep -f p3d_chain.sh > /dev/null || break
  sleep 30
done
echo "[$(date '+%T')] P3-D 采臂链已结束"

echo "[$(date '+%T')] 等权重 MSE 的 CPU 任务结束…"
for i in $(seq 1 90); do
  pgrep -f p3w_weight_mse.py > /dev/null || break
  sleep 20
done
echo "[$(date '+%T')] 权重 MSE 已结束（若未结束则由锁机制让路）"

echo "[$(date '+%T')] ===== 步骤 1：P4 ① 逐题 CPU 侧补跑 ====="
bash p4_item_cpu.sh
echo "[$(date '+%T')] 步骤 1 结束"

echo "[$(date '+%T')] ===== 步骤 2：P4 ⑤ ARC/HellaSwag 子集 ====="
bash p4_arc_hella_chain.sh
echo "[$(date '+%T')] 步骤 2 结束"

echo "########## 串行队列全部结束 $(date '+%F %T') ##########"
