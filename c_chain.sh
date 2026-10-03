#!/bin/bash
# 任务书 Task C：benchmark 统计 + tokenizer 口径核查。
#   1) 两个档位 × 四个任务重跑一次评测，**这次把逐题结果落盘**（此前只存了聚合指标，
#      无法做 paired bootstrap / McNemar；任务书 §4 要求逐题 correctness）；
#   2) 每个任务做 fp16 vs 均衡档的 paired bootstrap + McNemar（≥5000 次重采样）；
#   3) LAMBADA 另给 ΔNLL 的 paired bootstrap 与 PPL 变化（若逐题里带 ll/nll 字段）；
#   4) 跑 lm-eval tokenization protocol 核查（40 组边界用例）。
# 日志：c_chain.log
#
# 用法：setsid nohup bash c_chain.sh < /dev/null > /dev/null 2>&1 &

MODEL_DIR=/home/disk/models/rwkv7-2.9b
LOG=$MODEL_DIR/c_chain.log
PY_NPU=/home/disk/miniconda3/envs/npu22/bin/python
PY_EVAL=/home/disk/lmeval/bin/python
LIMIT=${LIMIT:-200}
TASKS=${TASKS:-"piqa arc_easy hellaswag lambada_openai"}
ITERS=${ITERS:-5000}

cd "$MODEL_DIR" || exit 1
exec >> "$LOG" 2>&1
source /home/disk/cann80base/ascend-toolkit/set_env.sh
export HF_HOME=/home/disk/hf_cache HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_TELEMETRY=1

echo
echo "########## Task C 开始 $(date '+%F %T') ｜ 子集 $LIMIT ｜ bootstrap $ITERS ##########"

for p in $(pgrep -f "bash $(basename "$0")"); do
  [ "$p" != "$$" ] && { echo "已有另一份在跑（pid $p），退出"; exit 1; }
done

for i in $(seq 1 480); do
  [ -f "$MODEL_DIR/experiment.lock" ] || break
  echo "  等待别的实验链释放 experiment.lock（$i）"
  sleep 60
done
[ -f "$MODEL_DIR/experiment.lock" ] && { echo "锁仍被占用，退出"; exit 1; }

bash start_service.sh stop
pkill -f p4_score_server.py 2>/dev/null; sleep 3
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
  echo "===== [$(date '+%T')] 档位 $name ====="
  pkill -f p4_score_server.py 2>/dev/null; sleep 3
  setsid nohup "$PY_NPU" p4_score_server.py --port 8100 "$@" \
      > "c_score_${name}.log" 2>&1 < /dev/null &
  for i in $(seq 1 36); do
    sleep 5
    grep -q "服务已就绪" "c_score_${name}.log" 2>/dev/null && break
  done
  tail -2 "c_score_${name}.log"
  for task in $TASKS; do
    [ -s "samples_${name}_${task}.jsonl" ] && { echo "  跳过 $task（已有逐题）"; continue; }
    echo "  [$(date '+%T')] $task"
    "$PY_EVAL" p4_run_eval.py --tasks "$task" --limit "$LIMIT" \
        --base-url http://127.0.0.1:8100 \
        --samples-out "samples_${name}_${task}.jsonl" \
        --out "c_eval_${name}_${task}.json" || echo "    评测失败（$name/$task）"
  done
}

echo "--- [1/3] 两档重跑评测（逐题落盘）$(date '+%T')"
run_tier fp16
run_tier balanced --plan-file plan_p1_top8.json
pkill -f p4_score_server.py 2>/dev/null

echo "--- [2/3] 配对统计 $(date '+%T')"
"$PY_NPU" - <<'PYEOF'
import json, os
subprocess = __import__("subprocess")
PY = "/home/disk/miniconda3/envs/npu22/bin/python"
model_dir = "/home/disk/models/rwkv7-2.9b"
merged = {"task": "C. Benchmark 统计与评测口径", "date": "2026-10-03", "tasks": {}}
for task in ("piqa", "arc_easy", "hellaswag", "lambada_openai"):
    a = os.path.join(model_dir, "samples_fp16_%s.jsonl" % task)
    b = os.path.join(model_dir, "samples_balanced_%s.jsonl" % task)
    if not (os.path.exists(a) and os.path.exists(b)):
        print("  跳过 %s（缺逐题文件）" % task)
        continue
    out = os.path.join(model_dir, "benchmark_stats_%s.json" % task)
    cmd = [PY, os.path.join(model_dir, "benchmark_paired_stats.py"),
           "--fp16", a, "--int8", b, "--task", task, "--iters", "5000", "--out", out]
    subprocess.run(cmd, check=False)
    if os.path.exists(out):
        with open(out) as fh:
            merged["tasks"][task] = json.load(fh)
with open(os.path.join(model_dir, "benchmark_paired_stats.json"), "w") as fh:
    json.dump(merged, fh, ensure_ascii=False, indent=2)
print("已写出 benchmark_paired_stats.json")
PYEOF

echo "--- [3/3] tokenization protocol 核查 $(date '+%T')"
"$PY_NPU" tokenizer_protocol_check.py \
    --out "$MODEL_DIR/tokenization_protocol_report.json" || echo "tokenizer 核查失败"

echo "--- 收尾 $(date '+%F %T')"
free -m | head -2
ls -l benchmark_paired_stats.json tokenization_protocol_report.json 2>/dev/null
rm -f "$MODEL_DIR/experiment.lock"
pkill -f experiment_guard.sh 2>/dev/null
trap - EXIT
sleep 2
bash start_service.sh 8000
echo "########## Task C 完成（网页服务已恢复）$(date '+%F %T') ##########"
