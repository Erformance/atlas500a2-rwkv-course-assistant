#!/bin/bash
# 服务健康看门狗：不响应就重启（配合 crontab 每 5 分钟跑一次）。
# 注意：一台设备同一时刻只能有一份模型在跑（每份占 5.5GB 内存），
# 所以发现异常实例时先清干净再启动。

MODEL_DIR=/home/disk/models/rwkv7-2.9b
PIDFILE=$MODEL_DIR/service.pid
LOG=$MODEL_DIR/watchdog.log
FAILFILE=$MODEL_DIR/watchdog.fail          # 连续探测失败次数
LOCKFILE=$MODEL_DIR/experiment.lock        # 存在则表示正在跑 NPU 实验，看门狗全程不介入
PY=/home/disk/miniconda3/envs/npu22/bin/python
MAX_FAIL=3                                  # 连续失败这么多次才重启（每 5 分钟一次 ≈ 15 分钟）

# 实验期间（experiment.lock 存在）完全不介入：实验脚本要独占 NPU，
# 服务被拉起来会与实验抢设备（表现为加载 .om 失败 145001）。
if [ -f "$LOCKFILE" ]; then
  exit 0
fi

alive() {
  [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null || return 1
  # 服务正在推理时不打扰（单线程服务此刻无法回应健康探测）
  [ -f "$MODEL_DIR/service.busy" ] && return 0
  "$PY" - <<'PYEOF' 2>/dev/null
import urllib.request
urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=45).read()
PYEOF
}

diagnose() {
  {
    echo "--- 探测失败时的现场 $(date '+%F %T')"
    echo "进程: $(ps -eo pid,stat,etime,pcpu,rss,args | grep '[r]wkv7_http' | head -2)"
    echo "端口: $(netstat -tlnp 2>/dev/null | grep 8000 | head -2)"
    echo "已建立连接: $(netstat -tn 2>/dev/null | grep -c ':8000')"
    echo "忙标记: $(ls -l "$MODEL_DIR/service.busy" 2>/dev/null || echo 无)"
    echo "负载: $(cat /proc/loadavg)"
  } >> "$LOG"
}

if alive; then
  echo 0 > "$FAILFILE"
  exit 0
fi

# 探不通不一定真挂了（可能刚好在长回答的尾巴上），隔 20 秒复探一次
sleep 20
if alive; then
  echo 0 > "$FAILFILE"
  echo "[$(date '+%F %T')] 首次探测失败但复探正常，跳过重启" >> "$LOG"
  exit 0
fi

FAILS=$(cat "$FAILFILE" 2>/dev/null || echo 0)
FAILS=$((FAILS + 1))
echo "$FAILS" > "$FAILFILE"
echo "[$(date '+%F %T')] 复探仍失败，连续第 $FAILS/$MAX_FAIL 次" >> "$LOG"
diagnose

if [ "$FAILS" -lt "$MAX_FAIL" ]; then
  exit 0                                  # 再观察一个周期，不急着杀
fi

echo "[$(date '+%F %T')] 连续 $FAILS 次无响应，执行重启" >> "$LOG"
[ -f "$PIDFILE" ] && kill -9 "$(cat "$PIDFILE")" 2>/dev/null
pkill -9 -f rwkv7_http.py 2>/dev/null
rm -f "$PIDFILE"
echo 0 > "$FAILFILE"
sleep 3
cd "$MODEL_DIR" || exit 1
bash start_service.sh 8000 >> "$LOG" 2>&1
echo "[$(date '+%F %T')] 重启完成" >> "$LOG"
