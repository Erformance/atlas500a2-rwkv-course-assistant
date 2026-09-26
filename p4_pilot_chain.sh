#!/bin/bash
# P4 试点：在真实基准上对比"精确档 fp16"与"均衡档 p1 top8"。
#   * 任务：piqa 与 lambada_openai，各取固定子集（这台设备 prefill 只有 5~12 tok/s，
#     完整基准要按天算，子集口径在报告里写清楚）
#   * 打分走 p4_score_server.py（与网页服务同一套引擎/同一份 .om/同一个 tokenizer）
#   * 每个档位跑一次 lm-eval（两个任务合并跑，省引擎加载与 harness 开销）
# 跑完恢复网页服务（是否关机由人决定）。日志：p4_pilot.log
#
# 用法：setsid nohup bash p4_pilot_chain.sh < /dev/null > /dev/null 2>&1 &

MODEL_DIR=/home/disk/models/rwkv7-2.9b
LOG=$MODEL_DIR/p4_pilot.log
PY_NPU=/home/disk/miniconda3/envs/npu22/bin/python
PY_EVAL=/home/disk/lmeval/bin/python
LIMIT=${LIMIT:-200}
TASKS=${TASKS:-piqa,lambada_openai}

cd "$MODEL_DIR" || exit 1
exec >> "$LOG" 2>&1
source /home/disk/cann80base/ascend-toolkit/set_env.sh
export HF_HOME=/home/disk/hf_cache HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_TELEMETRY=1
export PIP_CACHE_DIR=/home/disk/pip_cache

echo
echo "########## P4 试点开始 $(date '+%F %T') ｜ 任务 $TASKS ｜ 子集 $LIMIT ##########"

for p in $(pgrep -f "bash $(basename "$0")"); do
  [ "$p" != "$$" ] && { echo "已有另一份在跑（pid $p），退出"; exit 1; }
done

bash start_service.sh stop
pgrep -f p4_score_server.py > /dev/null && { pkill -f p4_score_server.py; sleep 3; }
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

run_tier() {   # $1=名字  $2=可选 --plan-file 参数
  local name="$1"; shift
  echo "===== [$(date '+%T')] 档位 $name ====="
  pkill -f p4_score_server.py 2>/dev/null
  sleep 3
  setsid nohup "$PY_NPU" p4_score_server.py --port 8100 "$@" \
      > "p4_score_${name}.log" 2>&1 < /dev/null &
  for i in $(seq 1 30); do
    sleep 5
    grep -q "服务已就绪" "p4_score_${name}.log" 2>/dev/null && break
  done
  tail -2 "p4_score_${name}.log"
  "$PY_EVAL" p4_run_eval.py --tasks "$TASKS" --limit "$LIMIT" \
      --base-url http://127.0.0.1:8100 --out "p4_eval_${name}.json"
}

run_tier fp16
run_tier balanced --plan-file plan_p1_top8.json

echo "--- 收尾 $(date '+%F %T')"
pkill -f p4_score_server.py 2>/dev/null
rm -f "$MODEL_DIR/experiment.lock"
pkill -f experiment_guard.sh 2>/dev/null
trap - EXIT
sleep 2
bash start_service.sh 8000
echo "########## P4 试点完成（网页服务已恢复）$(date '+%F %T') ##########"
