#!/bin/bash
# 启动课程助手常驻服务（模型只加载一次，之后每个请求直接推理）
#   bash start_service.sh          # 默认 8000 端口，档位读 service.tier
#   bash start_service.sh 8080     # 指定端口
#   bash start_service.sh stop     # 停止
#
# 换档：网页上直接选（服务退出码 75 → service_supervisor.sh 换档重启，约 45 秒）；
# 或在设备上：echo fast > service.tier && bash start_service.sh stop && bash start_service.sh

MODEL_DIR=/home/disk/models/rwkv7-2.9b
PORT=${1:-8000}
LOG=$MODEL_DIR/service.log
PIDFILE=$MODEL_DIR/service.pid

if [ "$1" = "stop" ]; then
  if [ -f "$PIDFILE" ]; then
    kill "$(cat "$PIDFILE")" 2>/dev/null && echo "已停止监督脚本 pid=$(cat "$PIDFILE")"
    rm -f "$PIDFILE"
  fi
  pkill -f "service_supervisor.sh" 2>/dev/null
  pkill -f "rwkv7_http.py" 2>/dev/null && echo "已停止服务"
  exit 0
fi

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "服务已在运行 pid=$(cat "$PIDFILE")"
  exit 0
fi

source /home/disk/cann80base/ascend-toolkit/set_env.sh
cd "$MODEL_DIR" || exit 1
# 先把上次的日志留一份（否则服务异常退出时的报错会被这次启动覆盖掉，没法排查），
# 再清空当前日志，避免把上一次的"就绪"当成这次启动成功的标志。
[ -s "$LOG" ] && cp -f "$LOG" "$LOG.prev"
: > "$LOG"
echo "[$(date '+%F %T')] 启动服务，端口 $PORT，档位 $(cat "$MODEL_DIR/service.tier" 2>/dev/null || echo "(默认)")" >> "$LOG"
# 由监督脚本托管：换档时服务以 75 退出，脚本用新档位重启
setsid nohup bash service_supervisor.sh "$PORT" >> "$LOG" 2>&1 < /dev/null &
echo $! > "$PIDFILE"

echo "正在加载模型（约 45 秒）……"
for i in $(seq 1 60); do
  sleep 2
  if grep -q "服务已就绪" "$LOG" 2>/dev/null; then
    echo "服务已就绪：http://192.168.31.50:$PORT/"
    grep -m1 "^\[档位\]" "$LOG" | sed 's/^/  /'
    echo "日志：tail -f $LOG"
    exit 0
  fi
done
echo "启动超时，请看日志：$LOG"
tail -n 20 "$LOG"
