#!/bin/bash
# 无人值守收尾：等 int8 编译（含 head）全部结束后自动关机。
# 日志写在 autopoweroff.log，明天上电后可以直接看结果。

LOG=/home/disk/models/rwkv7-2.9b/autopoweroff.log
MODEL_DIR=/home/disk/models/rwkv7-2.9b

echo "[$(date '+%F %T')] 看护启动：等待编译进程结束" >> "$LOG"

while ps -eo args | grep -q "[b]uild_int8_layers.sh"; do
  sleep 60
done

sleep 120     # 留两分钟给 atc 收尾写盘

{
  echo "[$(date '+%F %T')] 编译进程已结束"
  echo "int8 层数: $(ls "$MODEL_DIR"/layer*_q.om 2>/dev/null | wc -l)/32"
  ls -l "$MODEL_DIR"/head_q.om 2>&1
  echo "--- build_all.log 末尾 ---"
  tail -n 6 "$MODEL_DIR/build_all.log"
} >> "$LOG" 2>&1

sync
sleep 5
echo "[$(date '+%F %T')] 执行关机" >> "$LOG"
sync
systemctl --no-block poweroff || poweroff || /sbin/poweroff || shutdown -h now
