#!/bin/bash
# P4 ⑤：补齐协议 §6 要求的 ARC-Easy 与 HellaSwag 子集（此前只跑了 PIQA 与 LAMBADA）。
#   * 口径与 p4_pilot_chain.sh 完全一致：同一套打分服务（同 .om / 同 tokenizer）、
#     同一份 lm-evaluation-harness、同一子集大小（默认 200 题）；
#   * 两个档位：精确档 fp16 与均衡档 plan_p1_top8.json；
#   * 数据集走 hf-mirror（设备直连），首次运行会先下载 arc_easy / hellaswag。
# 日志：p4_arc_hella.log；产物 p4_evalfp16/balanced_{arc,hella}.json
#
# 用法：setsid nohup bash p4_arc_hella_chain.sh < /dev/null > /dev/null 2>&1 &

MODEL_DIR=/home/disk/models/rwkv7-2.9b
LOG=$MODEL_DIR/p4_arc_hella.log
PY_NPU=/home/disk/miniconda3/envs/npu22/bin/python
PY_EVAL=/home/disk/lmeval/bin/python
LIMIT=${LIMIT:-200}
TASKS=${TASKS:-arc_easy,hellaswag}
# 只跑指定档位：TIERS="balanced" 就只补均衡档（fp16 档 9-28 已跑完，见 p4_eval_fp16_arc_hella.json）
TIERS=${TIERS:-fp16 balanced}

cd "$MODEL_DIR" || exit 1
exec >> "$LOG" 2>&1
source /home/disk/cann80base/ascend-toolkit/set_env.sh
export HF_HOME=/home/disk/hf_cache HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_TELEMETRY=1
export PIP_CACHE_DIR=/home/disk/pip_cache

echo
echo "########## P4 ⑤ ARC/HellaSwag 子集开始 $(date '+%F %T') ｜ 任务 $TASKS ｜ 子集 $LIMIT ##########"

for p in $(pgrep -f "bash $(basename "$0")"); do
  [ "$p" != "$$" ] && { echo "已有另一份在跑（pid $p），退出"; exit 1; }
done

# 同时只允许一个重活：等别的实验链把锁交出来。
# 9-28 教训：原来"等 60 分钟就硬上"会让两条重活并发（CPU torch 6GB + NPU 引擎 5.5GB），
# 直接把设备压死。现在改成等满 8 小时就**退出**，绝不做并发。
for i in $(seq 1 480); do
  [ -f "$MODEL_DIR/experiment.lock" ] || break
  echo "  等待别的实验链释放 experiment.lock（$i）"
  sleep 60
done
if [ -f "$MODEL_DIR/experiment.lock" ]; then
  echo "锁仍被占用，本链退出（不做并发）"
  exit 1
fi

bash start_service.sh stop
pgrep -f p4_score_server.py > /dev/null && { pkill -f p4_score_server.py; sleep 3; }
for i in $(seq 1 60); do
  avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  [ "$avail" -gt 6000 ] && break
  sleep 10
done
echo "启动前可用内存 ${avail}MB"

touch "$MODEL_DIR/experiment.lock"
MINE=1
pgrep -f experiment_guard.sh > /dev/null || \
    setsid nohup bash experiment_guard.sh > /dev/null 2>&1 &
trap '[ "$MINE" = "1" ] && rm -f "$MODEL_DIR/experiment.lock"; pkill -f experiment_guard.sh' EXIT

run_tier() {   # $1=名字  $2..=打分服务参数
  local name="$1"; shift
  if [ -s "p4_eval_${name}_arc_hella.json" ]; then
    echo "===== 档位 $name 已有结果，跳过 ====="
    return 0
  fi
  echo "===== [$(date '+%T')] 档位 $name ====="
  pkill -f p4_score_server.py 2>/dev/null
  sleep 3
  setsid nohup "$PY_NPU" p4_score_server.py --port 8100 "$@" \
      > "p4_score_arc_${name}.log" 2>&1 < /dev/null &
  for i in $(seq 1 36); do
    sleep 5
    grep -q "服务已就绪" "p4_score_arc_${name}.log" 2>/dev/null && break
  done
  tail -2 "p4_score_arc_${name}.log"
  "$PY_EVAL" p4_run_eval.py --tasks "$TASKS" --limit "$LIMIT" \
      --base-url http://127.0.0.1:8100 --out "p4_eval_${name}_arc_hella.json" \
      || echo "评测失败（$name）"
}

for tier in $TIERS; do
  case "$tier" in
    fp16)     run_tier fp16 ;;
    balanced) run_tier balanced --plan-file plan_p1_top8.json ;;
    *)        echo "未知档位 $tier，跳过" ;;
  esac
done

echo "--- 收尾 $(date '+%F %T')"
pkill -f p4_score_server.py 2>/dev/null
rm -f "$MODEL_DIR/experiment.lock"
pkill -f experiment_guard.sh 2>/dev/null
trap - EXIT
sleep 2
bash start_service.sh 8000
echo "########## P4 ⑤ ARC/HellaSwag 完成（网页服务已恢复）$(date '+%F %T') ##########"
