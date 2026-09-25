#!/bin/bash
# P5 补充：真正的 2048 前缀（上一轮语料只有 1104 token，"2048" 那行实际喂的是 1104）。
# 用 PILOT_TEXT×8（2208 token）重测 4 个臂 × 2 次重复，结果单独落到 p5_bench_2048.json。
# 跑完恢复网页服务。
#
# 用法：setsid nohup bash p5b_chain.sh < /dev/null > /dev/null 2>&1 &

MODEL_DIR=/home/disk/models/rwkv7-2.9b
LOG=$MODEL_DIR/p5b_chain.log
PY=/home/disk/miniconda3/envs/npu22/bin/python

cd "$MODEL_DIR" || exit 1
exec >> "$LOG" 2>&1
source /home/disk/cann80base/ascend-toolkit/set_env.sh

echo
echo "########## P5 补充（真 2048 前缀）开始 $(date '+%F %T') ##########"

for p in $(pgrep -f "bash $(basename "$0")"); do
  [ "$p" != "$$" ] && { echo "已有另一份 $(basename "$0") 在跑（pid $p），退出"; exit 1; }
done

bash start_service.sh stop
for i in $(seq 1 24); do
  avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  [ "$avail" -gt 6000 ] && break
  sleep 5
done
echo "启动前可用内存 ${avail}MB"

touch "$MODEL_DIR/experiment.lock"
MINE=1
pgrep -f experiment_guard.sh > /dev/null || \
    setsid nohup bash experiment_guard.sh > /dev/null 2>&1 &
trap '[ "$MINE" = "1" ] && rm -f "$MODEL_DIR/experiment.lock"; pkill -f experiment_guard.sh' EXIT

"$PY" p5_bench.py --arms precise,balanced,fast,full_int8 \
    --prefixes 2048 --decode 128 --repeats 2 --corpus-repeat 8 \
    --parts-dir "$MODEL_DIR/p5_parts_2048" --out "$MODEL_DIR/p5_bench_2048.json" \
    || echo "补充测量有失败项"

echo "--- 收尾 $(date '+%F %T')"
rm -f "$MODEL_DIR/experiment.lock"
pkill -f experiment_guard.sh 2>/dev/null
trap - EXIT
sleep 2
bash start_service.sh 8000
echo "########## P5 补充完成（网页服务已恢复）$(date '+%F %T') ##########"
