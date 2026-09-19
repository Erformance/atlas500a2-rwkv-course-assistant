#!/bin/bash
# 开机自恢复：拉起 NPU 管理进程（slogd / dmp_daemon），再启动课程助手服务。
# 设备重启后 slogd 与 dmp_daemon 不会自启，不拉起的话 npu-smi 会报 -8010、NPU 不可用。

MODEL_DIR=/home/disk/models/rwkv7-2.9b
LOG=$MODEL_DIR/boot.log

echo "[$(date '+%F %T')] boot_all 开始" >> "$LOG"

# 1) NPU 管理进程
if ! npu-smi info > /dev/null 2>&1; then
  echo "[$(date '+%F %T')] npu-smi 不可用，拉起 slogd / dmp_daemon" >> "$LOG"
  pgrep -f "/var/slogd" > /dev/null || (setsid nohup /var/slogd >> "$LOG" 2>&1 &)
  sleep 2
  pgrep -f "dmp_daemon" > /dev/null || (setsid nohup /var/dmp_daemon -I -U 8087 >> "$LOG" 2>&1 &)
  sleep 5
fi
npu-smi info 2>&1 | sed -n '7p' >> "$LOG"

# 2) 课程助手服务
cd "$MODEL_DIR" || exit 1
bash start_service.sh "${1:-8000}" >> "$LOG" 2>&1
echo "[$(date '+%F %T')] boot_all 结束" >> "$LOG"
