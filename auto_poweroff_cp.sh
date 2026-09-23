#!/bin/bash
# 全栈重校准跑完自动关机：等 build_cp_layers.sh 退出 → 收尾记录 → 断电。
# 日志：autopoweroff_cp.log（明天上电后直接看这个文件即可判断昨晚结果）

MODEL_DIR=/home/disk/models/rwkv7-2.9b
LOG=$MODEL_DIR/autopoweroff_cp.log

echo "[$(date '+%F %T')] 看护启动：等待 build_cp_layers.sh 结束" >> "$LOG"

while ps -eo args | grep -q "[b]uild_cp_layers.sh"; do
  sleep 60
done

sleep 180          # 留三分钟给最后的 ATC 收尾与写盘

{
  echo "[$(date '+%F %T')] 重建进程已结束"
  echo "layer_*_cp_q.om 数量: $(ls "$MODEL_DIR"/layer*_cp_q.om 2>/dev/null | wc -l)/32"
  ls -l "$MODEL_DIR"/head_cp_q.om 2>&1 | tail -1
  echo "实验锁: $(ls "$MODEL_DIR"/experiment.lock 2>/dev/null || echo '已释放')"
  echo "--- build_cp.log 末尾 ---"
  tail -n 8 "$MODEL_DIR/build_cp.log"
  echo "--- watchdog.log 末尾 ---"
  tail -n 3 "$MODEL_DIR/watchdog.log" 2>/dev/null
} >> "$LOG" 2>&1

sync
sleep 5
echo "[$(date '+%F %T')] 执行关机" >> "$LOG"
sync
systemctl --no-block poweroff || poweroff || /sbin/poweroff
