#!/bin/bash
# 内存看门狗：Atlas 500 A2 只有 11GB 内存且无 swap，ATC 并发过高会把整机拖死。
# 可用内存连续 3 次（约 90s）低于 LIMIT_MB 时，杀掉最新的 atc.bin 进程保命。

LOG=/home/disk/models/rwkv7-2.9b/mem_guard.log
LIMIT_MB=${LIMIT_MB:-700}
low=0
while true; do
  avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  if [ "$avail" -lt "$LIMIT_MB" ]; then
    low=$((low + 1))
  else
    low=0
  fi
  if [ "$low" -ge 3 ]; then
    pid=$(ps -eo pid,args | grep '[a]tc.bin' | awk 'END{print $1}')
    echo "[$(date +%H:%M:%S)] 可用内存 ${avail}MB 持续偏低，杀 atc pid=${pid:-无}" >> "$LOG"
    [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null
    low=0
  fi
  sleep 30
done
