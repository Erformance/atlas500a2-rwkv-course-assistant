#!/bin/bash
# 收尾看护（2026-09-25 夜）：等 2048 补跑链结束 → 落现场 → 释放实验锁 → 强制断电。
# 说明：普通 poweroff 曾被 60s 硬件看门狗打断成重启，这里用 poweroff -f 缩短关机窗口。
# 日志：p5_wrapup.log

MODEL_DIR=/home/disk/models/rwkv7-2.9b
LOG=$MODEL_DIR/p5_wrapup.log

cd "$MODEL_DIR" || exit 1
exec >> "$LOG" 2>&1
echo "[$(date '+%F %T')] 收尾看护启动：等待 p5b_chain.sh 结束"

while pgrep -f "[p]5b_chain.sh" > /dev/null; do
  sleep 20
done
sleep 15

echo "[$(date '+%F %T')] 补跑链已结束，现场如下"
echo "--- 真 2048 补测结果 ---"
cat "$MODEL_DIR/p5_bench_2048.json" 2>/dev/null || echo "（p5_bench_2048.json 不存在）"
echo "--- 主矩阵（重新汇总后的中位数）---"
/home/disk/miniconda3/envs/npu22/bin/python - "$MODEL_DIR" <<'PYEOF' 2>/dev/null || true
import json, sys, os
d = json.load(open(os.path.join(sys.argv[1], "p5_bench.json")))
for arm, v in d["arms"].items():
    print("%-10s 加载 %5.1fs｜RSS %7.1fMB" % (arm, v["load_s_median"], v["rss_mb_median"]))
PYEOF
echo "--- 内存与 NPU ---"
free -m | head -2
npu-smi info 2>&1 | sed -n '7p'

rm -f "$MODEL_DIR/experiment.lock"
pkill -f experiment_guard.sh 2>/dev/null
sync
sleep 5
echo "[$(date '+%F %T')] 强制断电"
sync
poweroff -f
