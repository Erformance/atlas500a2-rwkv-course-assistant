#!/bin/bash
# P3-D 全链：H=128 窗口 + 跨分布（p1test）复核
#   1) collect：128 步前缀，33 个目标 × 2 份文本，参考臂 fp16 同长度
#   2) analyze：H=1/8/32/64/128 的排序相关、top-k 重合、配对 bootstrap、跨分布排序一致性
#   3) plan：H=8 选出的 8 层计划在 128 步窗口实测 KL，与线上 P1 计划同窗口对照
# 跑完恢复网页服务（不关机；关机由人决定）。
# 日志：p3d_chain.log
#
# 用法：setsid nohup bash p3d_chain.sh < /dev/null > /dev/null 2>&1 &

MODEL_DIR=/home/disk/models/rwkv7-2.9b
LOG=$MODEL_DIR/p3d_chain.log
PY=/home/disk/miniconda3/envs/npu22/bin/python

cd "$MODEL_DIR" || exit 1
exec >> "$LOG" 2>&1
source /home/disk/cann80base/ascend-toolkit/set_env.sh

echo
echo "########## P3-D（H=128 + 跨分布）开始 $(date '+%F %T') ##########"

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

"$PY" p3d_horizon.py --mode all || echo "P3-D 有失败项（可重跑续做）"

echo "--- 收尾 $(date '+%F %T')"
free -m | head -2
ls -l p3d_result.json 2>/dev/null
rm -f "$MODEL_DIR/experiment.lock"
pkill -f experiment_guard.sh 2>/dev/null
trap - EXIT
sleep 2
bash start_service.sh 8000
echo "########## P3-D 完成（网页服务已恢复）$(date '+%F %T') ##########"
