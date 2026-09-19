#!/bin/bash
# 服务健康看门狗：不响应就重启（配合 crontab 每 5 分钟跑一次）。
# 注意：一台设备同一时刻只能有一份模型在跑（每份占 5.5GB 内存），
# 所以发现异常实例时先清干净再启动。

MODEL_DIR=/home/disk/models/rwkv7-2.9b
PIDFILE=$MODEL_DIR/service.pid
LOG=$MODEL_DIR/watchdog.log
PY=/home/disk/miniconda3/envs/npu22/bin/python

alive() {
  [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null || return 1
  # 服务正在推理时不打扰（单线程服务此刻无法回应健康探测）
  [ -f "$MODEL_DIR/service.busy" ] && return 0
  "$PY" - <<'PYEOF' 2>/dev/null
import urllib.request
urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=30).read()
PYEOF
}

if alive; then
  exit 0
fi

# 一次探不通不一定真挂了，隔 15 秒再探一次，避免误杀
sleep 15
if alive; then
  echo "[$(date '+%F %T')] 首次探测失败但复探正常，跳过重启" >> "$LOG"
  exit 0
fi

echo "[$(date '+%F %T')] 服务未响应，准备重启" >> "$LOG"
[ -f "$PIDFILE" ] && kill -9 "$(cat "$PIDFILE")" 2>/dev/null
pkill -9 -f rwkv7_http.py 2>/dev/null
rm -f "$PIDFILE"
sleep 3
cd "$MODEL_DIR" || exit 1
bash start_service.sh 8000 >> "$LOG" 2>&1
echo "[$(date '+%F %T')] 重启完成" >> "$LOG"
