#!/bin/bash
# P4 ②（过夜任务）：把 RWKV7-G1j-1.5B 部署到 NPU，并跑与 2.9B 相同的 P4 子集，
# 用来检验"逐层混合精度 + 状态年龄校准"的结论是否**跨规模**成立。
#
# 流程：导出 24 层 ONNX → 逐层 ATC 编译（单 worker 串行）→ 导出/编译输出头 →
#       引擎自检（生成一小段中文）→ PIQA+LAMBADA 各 200 题（fp16）→ 同步 → 强制断电。
# 关键环境：RWKV_MODEL_DIR=/home/disk/models/rwkv7-1.5b（引擎维度从 config.json 读）。
# 日志：p4b_15b.log（各阶段另有自己的日志）。可中断续跑（已有 .om 会跳过）。
#
# 用法：setsid nohup bash p4b_15b_chain.sh < /dev/null > /dev/null 2>&1 &

MD=/home/disk/models/rwkv7-1.5b          # 1.5B 模型目录（也是产物目录）
SRC=/home/disk/models/rwkv7-2.9b         # 脚本来源（导出/编译/评测脚本都在这里）
LOG=$MD/p4b_15b.log
PY_EXPORT=/home/disk/miniconda3/envs/rwkv7/bin/python
PY_NPU=/home/disk/miniconda3/envs/npu22/bin/python
PY_EVAL=/home/disk/lmeval/bin/python
CANN_ENV=/home/disk/cann80base/ascend-toolkit/set_env.sh
TBE_PY=/home/disk/miniconda3/envs/npu22/bin
NLAYER=24
LIMIT=${LIMIT:-200}

export RWKV_MODEL_DIR=$MD
cd "$MD" || exit 1
exec >> "$LOG" 2>&1
source "$CANN_ENV"
export HF_HOME=/home/disk/hf_cache HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_TELEMETRY=1

echo
echo "########## P4 ② 1.5B 部署开始 $(date '+%F %T') ##########"

for p in $(pgrep -f "bash $(basename "$0")"); do
  [ "$p" != "$$" ] && { echo "已有另一份在跑，退出"; exit 1; }
done

bash "$SRC/start_service.sh" stop
for i in $(seq 1 24); do
  avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  [ "$avail" -gt 6000 ] && break
  sleep 5
done
echo "启动前可用内存 ${avail}MB"
touch "$SRC/experiment.lock"
MINE=1
pgrep -f experiment_guard.sh > /dev/null || \
    setsid nohup bash "$SRC/experiment_guard.sh" > /dev/null 2>&1 &
trap '[ "$MINE" = "1" ] && rm -f "$SRC/experiment.lock"; pkill -f experiment_guard.sh' EXIT

# ---- 1) 逐层导出 + 编译 -----------------------------------------------------
for i in $(seq 0 $((NLAYER - 1))); do
  om="$MD/layer${i}.om"
  if [ -f "$om" ]; then echo "[$(date +%H:%M:%S)] layer$i 跳过（已有）"; continue; fi
  echo "[$(date +%H:%M:%S)] === layer$i 导出"
  nice -n 19 "$PY_EXPORT" "$SRC/rwkv7_export_chunk.py" --model-dir "$MD" \
      --start "$i" --layers 1 --dtype float32 --out "$MD/layer${i}.onnx" \
      > "$MD/export_layer${i}.log" 2>&1
  if [ ! -f "$MD/layer${i}.onnx" ]; then echo "  layer$i 导出失败"; continue; fi
  echo "[$(date +%H:%M:%S)]   ATC 编译"
  PATH="$TBE_PY:$PATH" TE_PARALLEL_COMPILER=1 nice -n 19 atc \
      --model="$MD/layer${i}.onnx" --framework=5 --output="$MD/layer${i}" \
      --soc_version=Ascend310B1 --input_format=ND --log=error \
      > "$MD/atc_layer${i}.log" 2>&1
  [ -f "$om" ] && echo "[$(date +%H:%M:%S)] layer$i OK" || echo "[$(date +%H:%M:%S)] layer$i 编译失败"
done

# ---- 2) 输出头 -------------------------------------------------------------
if [ ! -f "$MD/head.om" ]; then
  echo "[$(date +%H:%M:%S)] === 输出头导出"
  nice -n 19 "$PY_EXPORT" "$SRC/rwkv7_export_head.py" --model-dir "$MD" \
      --out "$MD/head.onnx" > "$MD/export_head.log" 2>&1
  if [ -f "$MD/head.onnx" ]; then
    echo "[$(date +%H:%M:%S)]   ATC 编译（输出头较大，约 30~40 分钟）"
    PATH="$TBE_PY:$PATH" TE_PARALLEL_COMPILER=1 nice -n 19 atc \
        --model="$MD/head.onnx" --framework=5 --output="$MD/head" \
        --soc_version=Ascend310B1 --input_format=ND --log=error \
        > "$MD/atc_head.log" 2>&1
  fi
  [ -f "$MD/head.om" ] && echo "[$(date +%H:%M:%S)] head OK" || echo "[$(date +%H:%M:%S)] head 失败"
fi

# ---- 3) 引擎自检：生成一小段中文 -------------------------------------------
echo "--- [3/4] 引擎自检 $(date '+%T')"
free -m | head -2
printf '%s\n' "什么是熵？" "/quit" | "$PY_NPU" "$SRC/rwkv7_chat.py" --serve --tokens 40 \
    2>&1 | grep --line-buffered -vE '^\[(INFO|EVENT|WARNING)\]' | tail -8

# ---- 4) P4 子集评估（与 2.9B 同口径）---------------------------------------
echo "--- [4/4] PIQA+LAMBADA 子集评估 $(date '+%T')"
setsid nohup "$PY_NPU" "$SRC/p4_score_server.py" --port 8100 > "$MD/p4_score_1p5b.log" 2>&1 < /dev/null &
for i in $(seq 1 30); do
  sleep 5
  grep -q "服务已就绪" "$MD/p4_score_1p5b.log" 2>/dev/null && break
done
tail -2 "$MD/p4_score_1p5b.log"
"$PY_EVAL" "$SRC/p4_run_eval.py" --tasks piqa,lambada_openai --limit "$LIMIT" \
    --base-url http://127.0.0.1:8100 --out "$MD/p4_eval_1p5b.json" || echo "评估失败"
pkill -f p4_score_server.py 2>/dev/null

# ---- 收尾：留现场 → 释放锁 → 断电 ------------------------------------------
echo "--- 收尾 $(date '+%F %T')"
ls -l "$MD"/layer*.om 2>/dev/null | wc -l
ls -l "$MD/p4_eval_1p5b.json" 2>/dev/null
free -m | head -2
npu-smi info 2>&1 | sed -n '7p'
rm -f "$SRC/experiment.lock"
pkill -f experiment_guard.sh 2>/dev/null
trap - EXIT
sync
sleep 5
echo "########## P4 ② 完成，强制断电 $(date '+%F %T') ##########"
sync
poweroff -f
