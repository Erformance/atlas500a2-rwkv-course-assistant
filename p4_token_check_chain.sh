#!/bin/bash
# P4 ①（可行版）：CPU 与 NPU 的逐 token logp 对照。
#   NPU 侧：打分服务的 /loglikelihood_detail（fp16 引擎）
#   CPU 侧：PyTorch 参考（bf16、关 MKLDNN，因为 aarch64 上 MKLDNN 不支持 bf16 matmul）
# 两侧都只算"同一前缀 + 前 4 个续写 token"，逐 token 比 logp（比只比求和更严格）。
# 用两道题：PIQA 第 0 题的第 1 个候选 + 一个短合成句。
# 日志：p4_token_check.log；产物 p4_token_npu.json / p4_token_cpu.json / p4_token_vs.json
#
# 用法：setsid nohup bash p4_token_check_chain.sh < /dev/null > /dev/null 2>&1 &

MODEL_DIR=/home/disk/models/rwkv7-2.9b
LOG=$MODEL_DIR/p4_token_check.log
PY_NPU=/home/disk/miniconda3/envs/npu22/bin/python
PY_CPU=/home/disk/miniconda3/envs/rwkv7/bin/python
PREFIX="Question: How do you close a door?
Answer:"
CONT1=" Turn the handle and push."
CTX2="The capital of France is"
CONT2=" Paris"

cd "$MODEL_DIR" || exit 1
exec >> "$LOG" 2>&1
source /home/disk/cann80base/ascend-toolkit/set_env.sh

echo
echo "########## P4 ① 逐 token 对照开始 $(date '+%F %T') ##########"

for p in $(pgrep -f "bash $(basename "$0")"); do
  [ "$p" != "$$" ] && { echo "已有另一份在跑，退出"; exit 1; }
done

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

echo "--- [1/3] NPU 侧 $(date '+%T')"
setsid nohup "$PY_NPU" p4_score_server.py --port 8100 > p4_score_token_check.log 2>&1 < /dev/null &
for i in $(seq 1 30); do
  sleep 5
  grep -q "服务已就绪" p4_score_token_check.log 2>/dev/null && break
done
tail -2 p4_score_token_check.log
"$PY_NPU" p4_token_check.py --mode npu --ctx "$PREFIX" --cont "$CONT1" \
    --max-tokens 4 --out p4_token_npu.json || echo "NPU 侧失败"
"$PY_NPU" p4_token_check.py --mode npu --ctx "$CTX2" --cont "$CONT2" \
    --max-tokens 2 --out p4_token_npu_b.json || echo "NPU 侧(短句)失败"
pkill -f p4_score_server.py; sleep 5

echo "--- [2/3] CPU 侧（bf16、关 MKLDNN，慢）$(date '+%T')"
for i in $(seq 1 24); do
  avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  [ "$avail" -gt 6500 ] && break
  sleep 5
done
echo "CPU 前可用内存 ${avail}MB"
"$PY_CPU" p4_token_check.py --mode cpu --ctx "$PREFIX" --cont "$CONT1" \
    --max-tokens 4 --out p4_token_cpu.json || echo "CPU 侧失败"
"$PY_CPU" p4_token_check.py --mode cpu --ctx "$CTX2" --cont "$CONT2" \
    --max-tokens 2 --out p4_token_cpu_b.json || echo "CPU 侧(短句)失败"

echo "--- [3/3] 比对 $(date '+%T')"
"$PY_NPU" p4_token_check.py --mode compare --compare p4_token_npu.json p4_token_cpu.json \
    --out p4_token_vs.json || echo "比对失败"
"$PY_NPU" p4_token_check.py --mode compare --compare p4_token_npu_b.json p4_token_cpu_b.json \
    --out p4_token_vs_b.json || echo "比对(短句)失败"

rm -f "$MODEL_DIR/experiment.lock"
pkill -f experiment_guard.sh 2>/dev/null
trap - EXIT
free -m | head -2
bash start_service.sh 8000
echo "########## P4 ① 完成（网页服务已恢复）$(date '+%F %T') ##########"
