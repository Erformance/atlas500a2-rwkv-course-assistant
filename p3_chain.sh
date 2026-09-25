#!/bin/bash
# P3（窄版）：测量窗口 H 的增量价值——逐层臂采集 + 离线排序分析 + 计划级实测。
# 跑完恢复网页服务（不关机）。带单实例检查与内存等待（同 P2 的教训）。
#
# 日志：p3_chain.log；产物 p3_result.json（+ p3_parts/ 里的逐目标 logits 可续跑）
#
# 用法：setsid nohup bash p3_chain.sh < /dev/null > /dev/null 2>&1 &

MODEL_DIR=/home/disk/models/rwkv7-2.9b
LOG=$MODEL_DIR/p3_chain.log
PY=/home/disk/miniconda3/envs/npu22/bin/python

cd "$MODEL_DIR" || exit 1
exec >> "$LOG" 2>&1
source /home/disk/cann80base/ascend-toolkit/set_env.sh

echo
echo "########## P3 开始 $(date '+%F %T') ##########"

for p in $(pgrep -f "bash $(basename "$0")"); do
  [ "$p" != "$$" ] && { echo "已有另一份 $(basename "$0") 在跑（pid $p），退出"; exit 1; }
done

bash start_service.sh stop
for i in $(seq 1 24); do
  avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  [ "$avail" -gt 6000 ] && break
  sleep 5
done
echo "启动前可用内存 ${avail}MB"

touch "$MODEL_DIR/experiment.lock"
MINE=1
pgrep -f experiment_guard.sh > /dev/null || \
    setsid nohup bash experiment_guard.sh > /dev/null 2>&1 &
trap '[ "$MINE" = "1" ] && rm -f "$MODEL_DIR/experiment.lock"; pkill -f experiment_guard.sh' EXIT

free -m | head -2
echo "--- [1/3] 采集 33 个逐层臂 $(date '+%T')"
"$PY" p3_horizon.py --mode collect || echo "采集阶段有失败项（见日志；可重跑续做）"

echo "--- [2/3] 离线分析（各 H 的排序相关性）$(date '+%T')"
"$PY" p3_horizon.py --mode analyze || echo "分析失败"

echo "--- [3/3] 计划级实测 $(date '+%T')"
"$PY" p3_horizon.py --mode plan || echo "计划级验证失败"

echo "--- 收尾 $(date '+%F %T')"
free -m | head -2
cat p3_result.json 2>/dev/null | head -60
rm -f "$MODEL_DIR/experiment.lock"
pkill -f experiment_guard.sh 2>/dev/null
trap - EXIT
sleep 2
bash start_service.sh 8000
echo "########## P3 完成（网页服务已恢复）$(date '+%F %T') ##########"
