#!/bin/bash
# 课程助手服务的"档位监督"：服务以退出码 75 退出 = 用户在网页上换了档，
# 这里就用 service.tier 里的新档位把它重新拉起来（加载 .om 约 45 秒）。
# 其它退出码按真故障处理：存活够久的重启，连续快速失败两次就停手，
# 交给 5 分钟健康看门狗去决定要不要再来（避免坏档位导致反复重启）。
#
#   setsid nohup bash service_supervisor.sh 8000 > /dev/null 2>&1 &
#
# 停止：`bash start_service.sh stop`（会先杀这个监督脚本，再杀服务进程）

MODEL_DIR=/home/disk/models/rwkv7-2.9b
PORT=${1:-8000}
LOG=$MODEL_DIR/service.log
PY=/home/disk/miniconda3/envs/npu22/bin/python

cd "$MODEL_DIR" || exit 1
source /home/disk/cann80base/ascend-toolkit/set_env.sh

fail=0
while :; do
  start=$(date +%s)
  "$PY" rwkv7_http.py --port "$PORT" >> "$LOG" 2>&1 < /dev/null
  code=$?
  dur=$(( $(date +%s) - start ))

  if [ "$code" -eq 75 ]; then
    echo "[$(date '+%F %T')] 换档重启（tier=$(cat "$MODEL_DIR/service.tier" 2>/dev/null)）" >> "$LOG"
    fail=0
    sleep 2
    continue
  fi

  if [ "$dur" -lt 60 ]; then
    fail=$((fail + 1))
    echo "[$(date '+%F %T')] 服务异常退出（code=$code，只活了 ${dur}s），连续第 $fail 次" >> "$LOG"
    if [ "$fail" -ge 2 ]; then
      echo "[$(date '+%F %T')] 连续快速失败，监督脚本停止（等健康看门狗处理）" >> "$LOG"
      rm -f "$MODEL_DIR/service.pid"
      break
    fi
  else
    fail=0
    echo "[$(date '+%F %T')] 服务退出（code=$code，活了 ${dur}s），重新拉起" >> "$LOG"
  fi
  sleep 3
done
