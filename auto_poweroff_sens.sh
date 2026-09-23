#!/bin/bash
# 敏感度测量跑完自动关机：
#   1) 等 sensitivity.py 退出；2) 记录结果摘要；3) **释放实验锁**（否则下次开机自启不会拉起服务）；
#   4) sync 后断电。日志：autopoweroff_sens.log

MODEL_DIR=/home/disk/models/rwkv7-2.9b
LOG=$MODEL_DIR/autopoweroff_sens.log
PY=/home/disk/miniconda3/envs/npu22/bin/python

echo "[$(date '+%F %T')] 看护启动：等待 sensitivity.py 结束" >> "$LOG"

while ps -eo args | grep -q "[s]ensitivity.py"; do
  sleep 60
done

sleep 120

{
  echo "[$(date '+%F %T')] 敏感度测量已结束"
  "$PY" - <<'PYEOF'
import json, os
p = "/home/disk/models/rwkv7-2.9b/sensitivity.json"
d = json.load(open(p))
tg = d["targets"]
print("已完成目标数: %d/33" % len(tg))
ranked = sorted(tg.items(), key=lambda kv: -kv[1]["kl_mean"])
print("最敏感的 6 个目标：")
for name, v in ranked[:6]:
    print("  %-6s KL %7.3f ｜ top-1 %5.1f%%" % (name, v["kl_mean"], v["top1_agreement"] * 100))
safe = [n for n, v in tg.items() if v["kl_mean"] < 0.02]
print("KL<0.02 的安全目标数: %d（%s…）" % (len(safe), ",".join(safe[:8])))
PYEOF
  echo "--- sensitivity.log 末尾 ---"
  tail -n 8 "$MODEL_DIR/sensitivity.log"
} >> "$LOG" 2>&1

# 关键：释放实验锁，保证下次开机自启能正常拉起 NPU 管理进程与网页服务
rm -f "$MODEL_DIR/experiment.lock"
echo "[$(date '+%F %T')] 已释放实验锁" >> "$LOG"

sync
sleep 5
echo "[$(date '+%F %T')] 执行关机" >> "$LOG"
sync
systemctl --no-block poweroff || poweroff || /sbin/poweroff
