#!/bin/bash
# 启动课程助手常驻服务（模型只加载一次，之后每个请求直接推理）
#   bash start_service.sh          # 默认 8000 端口
#   bash start_service.sh 8080     # 指定端口
#   bash start_service.sh stop     # 停止

MODEL_DIR=/home/disk/models/rwkv7-2.9b
PORT=${1:-8000}
LOG=$MODEL_DIR/service.log
PIDFILE=$MODEL_DIR/service.pid

if [ "$1" = "stop" ]; then
  if [ -f "$PIDFILE" ]; then
    kill "$(cat "$PIDFILE")" 2>/dev/null && echo "已停止服务 pid=$(cat "$PIDFILE")"
    rm -f "$PIDFILE"
  else
    pkill -f "rwkv7_http.py" && echo "已停止服务"
  fi
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
echo "[$(date '+%F %T')] 启动服务，端口 $PORT" >> "$LOG"
setsid nohup /home/disk/miniconda3/envs/npu22/bin/python rwkv7_http.py --port "$PORT" \
    >> "$LOG" 2>&1 < /dev/null &
echo $! > "$PIDFILE"

echo "正在加载模型（约 45 秒）……"
for i in $(seq 1 60); do
  sleep 2
  if grep -q "服务已就绪" "$LOG" 2>/dev/null; then
    echo "服务已就绪：http://192.168.31.50:$PORT/"
    echo "日志：tail -f $LOG"
    exit 0
  fi
done
echo "启动超时，请看日志：$LOG"
tail -n 20 "$LOG"
