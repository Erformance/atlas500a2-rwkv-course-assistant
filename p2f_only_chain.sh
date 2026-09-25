#!/bin/bash
# 只重跑 P2-F（P2-D 的结果已经有了）：修好 logits 对齐与速度口径之后重跑一遍。
# 跑完恢复网页服务，不关机。
#
# 这条链特意加了两个保护（10:00 那次翻车的教训）：
#   1) 单实例检查：两份链同时跑会互相删实验锁 → 看门狗把服务拉回来 → 两份模型抢内存 → 245000；
#   2) 停服务后等可用内存回到 6GB 以上再开实验（一份 fp16 引擎约 5.5GB）。
#
# 用法：setsid nohup bash p2f_only_chain.sh < /dev/null > /dev/null 2>&1 &

MODEL_DIR=/home/disk/models/rwkv7-2.9b
LOG=$MODEL_DIR/p2f_chain.log
PY=/home/disk/miniconda3/envs/npu22/bin/python

cd "$MODEL_DIR" || exit 1
exec >> "$LOG" 2>&1
source /home/disk/cann80base/ascend-toolkit/set_env.sh

echo
echo "########## P2-F 重跑开始 $(date '+%F %T') ##########"

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
pgrep -f experiment_guard.sh > /dev/null || \
    setsid nohup bash experiment_guard.sh > /dev/null 2>&1 &
trap '[ "$MINE" = "1" ] && rm -f "$MODEL_DIR/experiment.lock"; pkill -f experiment_guard.sh' EXIT
MINE=1

free -m | head -2
rm -rf p2f_parts
"$PY" p2f_free_gen.py --prefix 64 --tokens 64 --out p2f_result.json \
    || echo "P2-F 有失败项（见日志与 p2f_parts/）"

echo "--- 收尾 $(date '+%F %T')"
free -m | head -2
cat p2f_result.json 2>/dev/null | head -40
rm -f "$MODEL_DIR/experiment.lock"
pkill -f experiment_guard.sh 2>/dev/null
trap - EXIT
sleep 2
bash start_service.sh 8000
echo "########## P2-F 重跑完成 $(date '+%F %T') ##########"
