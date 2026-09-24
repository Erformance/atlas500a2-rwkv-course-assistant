#!/bin/bash
# 实验内存看门狗：跑 NPU 实验时启用，防止实验进程把内存压满导致整机假死。
#
# 逻辑：每 30 秒看一次 MemAvailable；连续 2 次低于 LIMIT_MB 时，
# 杀掉"模型目录下最新的 python 进程"（排除常驻服务 rwkv7_http.py），并记日志。
#
# 用法: setsid nohup bash experiment_guard.sh > /dev/null 2>&1 &
#       pkill -f experiment_guard.sh      # 实验结束后停掉

MODEL_DIR=/home/disk/models/rwkv7-2.9b
LOG=$MODEL_DIR/experiment_guard.log
LIMIT_MB=${LIMIT_MB:-900}

low=0
while true; do
  avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  if [ "$avail" -lt "$LIMIT_MB" ]; then
    low=$((low + 1))
  else
    low=0
  fi
  if [ "$low" -ge 2 ]; then
    pid=$(ps -eo pid,args | grep "[m]odels/rwkv7-2.9b" | grep -v "rwkv7_http.py" \
          | awk 'END{print $1}')
    echo "[$(date '+%F %T')] 可用内存 ${avail}MB 持续偏低，杀掉实验进程 ${pid:-无}" >> "$LOG"
    [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null
    low=0
    sleep 60                      # 杀完之后留时间回收内存
  fi
  sleep 30
done
