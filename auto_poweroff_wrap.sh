#!/bin/bash
# 收尾关机链（本次会话用）：等三档复核跑完 → 留现场 → 停看门狗 → 释放实验锁 → sync → 关机。
# 结果日志：autopoweroff_wrap.log（复核正文在 quality_fix.log，掉电后仍在盘上）

MODEL_DIR=/home/disk/models/rwkv7-2.9b
LOG=$MODEL_DIR/autopoweroff_wrap.log

exec >> "$LOG" 2>&1
echo "[$(date '+%F %T')] 收尾链启动：等待 quality_cmp.sh / rwkv7_chat.py 结束"

while pgrep -f "[q]uality_cmp.sh" > /dev/null || pgrep -f "[r]wkv7_chat.py" > /dev/null; do
  sleep 30
done

sleep 30

echo "[$(date '+%F %T')] 复核脚本已结束"
echo "--- quality_fix.log 全文 ---"
cat "$MODEL_DIR/quality_fix.log" 2>/dev/null
echo
echo "--- 内存 ---"
free -m
echo "--- NPU ---"
npu-smi info 2>&1 | sed -n '7p'

# 关掉 5 分钟健康看门狗（crond）与内存看门狗，避免关机窗口里又把服务拉起来抢 NPU
pkill -f experiment_guard.sh 2>/dev/null
systemctl stop crond 2>/dev/null || service crond stop 2>/dev/null
sleep 2

# 关键：释放实验锁，保证下次开机自启能正常拉起网页服务
rm -f "$MODEL_DIR/experiment.lock"
echo "[$(date '+%F %T')] 已停看门狗、释放实验锁"

sync
sleep 5
echo "[$(date '+%F %T')] 执行关机"
sync
systemctl --no-block poweroff || poweroff || /sbin/poweroff
