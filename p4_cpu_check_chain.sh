#!/bin/bash
# P4 ①：CPU(PyTorch) 与 NPU(OM) 的逐题对照（协议 §6）。
# 顺序：导出题目 → 打分服务(NPU)算 logp → 停服务 → PyTorch CPU 算同一批 → 比对 → 恢复服务。
# 注意：NPU 引擎(5.5G) 与 CPU bf16 权重(5.9G) 不能同时驻留，所以必须分两步。
# 日志：p4_cpu_check.log；产物 p4_items.json / p4_npu_scores.json / p4_cpu_scores.json / p4_cpu_vs_npu.json
#
# 用法：setsid nohup bash p4_cpu_check_chain.sh < /dev/null > /dev/null 2>&1 &

MODEL_DIR=/home/disk/models/rwkv7-2.9b
LOG=$MODEL_DIR/p4_cpu_check.log
PY_NPU=/home/disk/miniconda3/envs/npu22/bin/python
PY_CPU=/home/disk/miniconda3/envs/rwkv7/bin/python
PY_EVAL=/home/disk/lmeval/bin/python
ITEMS=${ITEMS:-3}

cd "$MODEL_DIR" || exit 1
exec >> "$LOG" 2>&1
source /home/disk/cann80base/ascend-toolkit/set_env.sh     # 打分服务要 import acl
export HF_HOME=/home/disk/hf_cache HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_TELEMETRY=1

echo
echo "########## P4 ① CPU↔NPU 逐题对照开始 $(date '+%F %T') ##########"

for p in $(pgrep -f "bash $(basename "$0")"); do
  [ "$p" != "$$" ] && { echo "已有另一份在跑，退出"; exit 1; }
done

# 1) 导出题目（不需要 NPU）
"$PY_EVAL" p4_dump_items.py --items "$ITEMS" --out p4_items.json || exit 1

bash start_service.sh stop
pkill -f p4_score_server.py 2>/dev/null; sleep 3
for i in $(seq 1 24); do
  avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  [ "$avail" -gt 6000 ] && break
  sleep 5
done
touch "$MODEL_DIR/experiment.lock"
MINE=1
pgrep -f experiment_guard.sh > /dev/null || \
    setsid nohup bash experiment_guard.sh > /dev/null 2>&1 &
trap '[ "$MINE" = "1" ] && rm -f "$MODEL_DIR/experiment.lock"; pkill -f experiment_guard.sh' EXIT

# 2) NPU 侧打分
echo "--- [1/3] NPU（fp16 引擎）算 logp $(date '+%T')"
setsid nohup "$PY_NPU" p4_score_server.py --port 8100 > p4_score_fp16_check.log 2>&1 < /dev/null &
for i in $(seq 1 30); do
  sleep 5
  grep -q "服务已就绪" p4_score_fp16_check.log 2>/dev/null && break
done
tail -2 p4_score_fp16_check.log
"$PY_NPU" p4_score_items.py --mode npu --items p4_items.json \
    --base-url http://127.0.0.1:8100 --out p4_npu_scores.json || echo "NPU 打分失败"
pkill -f p4_score_server.py; sleep 5

# 3) CPU 侧打分（把 NPU 引擎放掉之后再加载 PyTorch 权重）
echo "--- [2/3] CPU（PyTorch 参考）算同一批 $(date '+%T')"
free -m | head -2
for i in $(seq 1 24); do
  avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  [ "$avail" -gt 6500 ] && break
  sleep 5
done
echo "CPU 打分前可用内存 ${avail}MB"
"$PY_CPU" p4_score_items.py --mode cpu --model-dir "$MODEL_DIR" --dtype bfloat16 \
    --items p4_items.json --out p4_cpu_scores.json || echo "CPU 打分失败"

# 4) 比对
echo "--- [3/3] 比对 $(date '+%T')"
"$PY_NPU" p4_compare_items.py --npu p4_npu_scores.json --cpu p4_cpu_scores.json \
    --out p4_cpu_vs_npu.json || echo "比对失败"

rm -f "$MODEL_DIR/experiment.lock"
pkill -f experiment_guard.sh 2>/dev/null
trap - EXIT
free -m | head -2
bash start_service.sh 8000
echo "########## P4 ① 完成（网页服务已恢复）$(date '+%F %T') ##########"
