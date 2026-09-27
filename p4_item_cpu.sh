#!/bin/bash
# P4 ① 收尾：CPU(PyTorch) 侧对同一批 PIQA 题算 logp，并与已落盘的 NPU 结果比对。
# 复用 p4_cpu_check_chain.sh 的 [3]/[4] 两步——NPU 侧（p4_npu_scores.json）已经跑完，
# 上次是在 CPU 加载权重时被关机打断的，所以这里只补 CPU 与比对。
# 产物：p4_cpu_scores.json、p4_cpu_vs_npu.json；日志 p4_cpu_check2.log
#
# 用法：setsid nohup bash p4_item_cpu.sh < /dev/null > /dev/null 2>&1 &

MODEL_DIR=/home/disk/models/rwkv7-2.9b
LOG=$MODEL_DIR/p4_cpu_check2.log
PY_NPU=/home/disk/miniconda3/envs/npu22/bin/python
PY_CPU=/home/disk/miniconda3/envs/rwkv7/bin/python

cd "$MODEL_DIR" || exit 1
exec >> "$LOG" 2>&1

echo
echo "########## P4 ① CPU 侧补跑开始 $(date '+%F %T') ##########"

for p in $(pgrep -f "bash $(basename "$0")"); do
  [ "$p" != "$$" ] && { echo "已有另一份在跑（pid $p），退出"; exit 1; }
done

bash start_service.sh stop
for i in $(seq 1 60); do
  avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  [ "$avail" -gt 6500 ] && break
  sleep 10
done
echo "CPU 打分前可用内存 ${avail}MB"

touch "$MODEL_DIR/experiment.lock"
MINE=1
pgrep -f experiment_guard.sh > /dev/null || \
    setsid nohup bash experiment_guard.sh > /dev/null 2>&1 &
trap '[ "$MINE" = "1" ] && rm -f "$MODEL_DIR/experiment.lock"; pkill -f experiment_guard.sh' EXIT

echo "--- CPU（PyTorch 参考）算同一批 $(date '+%T')"
"$PY_CPU" p4_score_items.py --mode cpu --model-dir "$MODEL_DIR" --dtype bfloat16 \
    --items p4_items.json --out p4_cpu_scores.json || echo "CPU 打分失败"

echo "--- 比对 $(date '+%T')"
"$PY_NPU" p4_compare_items.py --npu p4_npu_scores.json --cpu p4_cpu_scores.json \
    --out p4_cpu_vs_npu.json || echo "比对失败"

echo "--- 收尾 $(date '+%F %T')"
free -m | head -2
rm -f "$MODEL_DIR/experiment.lock"
pkill -f experiment_guard.sh 2>/dev/null
trap - EXIT
sleep 2
bash start_service.sh 8000
echo "########## P4 ① CPU 侧补跑完成 $(date '+%F %T') ##########"
