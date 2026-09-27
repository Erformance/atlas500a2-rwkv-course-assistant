#!/bin/bash
# 收尾关机守护：等三条链全部结束 → 留现场（产物清单 + 内存/NPU 快照）→
# 停看门狗、释放实验锁 → sync → 关机（poweroff -f 三重试）。
# 日志：final_poweroff.log（掉电后仍在盘上）
#
# 用法：setsid nohup bash final_poweroff.sh < /dev/null > /dev/null 2>&1 &

MODEL_DIR=/home/disk/models/rwkv7-2.9b
LOG=$MODEL_DIR/final_poweroff.log

cd "$MODEL_DIR" || exit 1
exec >> "$LOG" 2>&1

echo
echo "########## 收尾关机守护启动 $(date '+%F %T') ##########"

# 等队列与尾链都结束（P3-E 跑在 queue_tail.sh 里）
for i in $(seq 1 4000); do
  if ! pgrep -f 'queue_next.sh' > /dev/null \
     && ! pgrep -f 'queue_tail.sh' > /dev/null \
     && ! pgrep -f 'p3e_chain.sh' > /dev/null \
     && ! pgrep -f 'p4_arc_hella_chain.sh' > /dev/null \
     && ! pgrep -f 'p4_item_cpu.sh' > /dev/null \
     && ! pgrep -f 'p3d_chain.sh' > /dev/null; then
    break
  fi
  sleep 30
done
sleep 60

echo
echo "--- 队列已结束，留现场 $(date '+%F %T') ---"
free -m | head -3
uptime
echo "--- 本轮产物 ---"
ls -l p3d_result.json p3w_weight_mse.json p3w_compare.json \
      p4_cpu_scores.json p4_cpu_vs_npu.json \
      p4_eval_*_arc_hella.json p3e_gramian.json 2>&1 | tail -12
echo "--- 各链日志尾部 ---"
for f in p3d_chain.log queue_next.log p4_cpu_check2.log p4_arc_hella.log queue_tail.log p3e_chain.log; do
  echo "== $f"
  tail -4 "$f" 2>/dev/null
done

# 停看门狗，避免关机窗口里 crond 又把网页服务拉起来
pkill -f experiment_guard.sh 2>/dev/null
bash start_service.sh stop 2>/dev/null
systemctl stop crond 2>/dev/null || service crond stop 2>/dev/null
sleep 2
rm -f "$MODEL_DIR/experiment.lock"
echo "[$(date '+%F %T')] 已停看门狗、释放实验锁"

sync
sleep 5
for i in 1 2 3; do
  echo "[$(date '+%F %T')] 第 $i 次关机尝试"
  sync
  /sbin/poweroff -f
  sleep 45
done
echo "[$(date '+%F %T')] 三次 poweroff -f 均未生效（需要人工处理）"
