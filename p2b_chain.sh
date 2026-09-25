#!/bin/bash
# P2-B 严格对照：在**样本数相同**的前提下，只改"状态年龄覆盖"这一个因素。
#   B0：32 组样本，全部落在年龄 0（32 段各采 1 次）—— 覆盖面广、轨迹短
#   B1：32 组样本，年龄 0 与 256（16 段各采 2 次）—— 覆盖面窄、轨迹长
# 另有两条参照臂（不需重编译）：_cp_q（旧 16 组、单文本开头采样）、_p1_q（96 组、0/64/256）。
# 只重编译 p1 top8 计划用到的那 8 层、不建 head，评估口径与其余实验一致（pilot + p1test）。
# 跑完自动强制断电。
#
# 日志：p2b_chain.log；结果：p2b_compare_pilot.json / p2b_compare_p1test.json
# 用法：setsid nohup bash p2b_chain.sh < /dev/null > /dev/null 2>&1 &

MODEL_DIR=/home/disk/models/rwkv7-2.9b
LOG=$MODEL_DIR/p2b_chain.log
PY=/home/disk/miniconda3/envs/npu22/bin/python
LAYERS="24 6 11 22 23 20 25 28"

cd "$MODEL_DIR" || exit 1
exec >> "$LOG" 2>&1
source /home/disk/cann80base/ascend-toolkit/set_env.sh

echo
echo "########## P2-B 开始 $(date '+%F %T') ##########"

# 单实例保护（两份链同时跑会互相删锁、把服务引回来抢内存 → 245000）
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
free -m | head -2

# 1) 两组等样本数校准数据（各 32 组）
if [ ! -f "$MODEL_DIR/calib_b0/layer00.npz" ]; then
  echo "--- [1/4] 采集 B0（32 段 × 年龄 0）$(date '+%T')"
  "$PY" capture_calib_p1.py --split calib --limit 32 --ages 0 \
      --out "$MODEL_DIR/calib_b0_raw" --max-samples 32 \
      --manifest "$MODEL_DIR/p2b_b0_manifest.json" || exit 1
  "$PY" pack_calib_p1.py --raw "$MODEL_DIR/calib_b0_raw" --out "$MODEL_DIR/calib_b0" || exit 1
fi
if [ ! -f "$MODEL_DIR/calib_b1/layer00.npz" ]; then
  echo "--- [1/4] 采集 B1（16 段 × 年龄 0/256）$(date '+%T')"
  "$PY" capture_calib_p1.py --split calib --limit 16 --ages 0,256 \
      --out "$MODEL_DIR/calib_b1_raw" --max-samples 32 \
      --manifest "$MODEL_DIR/p2b_b1_manifest.json" || exit 1
  "$PY" pack_calib_p1.py --raw "$MODEL_DIR/calib_b1_raw" --out "$MODEL_DIR/calib_b1" || exit 1
fi

# 2) 三条臂的计划文件（8 层子集，与 p1 top8 同一组层）
"$PY" - <<'PYEOF'
import json
layers = [24, 6, 11, 22, 23, 20, 25, 28]
for arm in ("b0", "b1"):
    with open("plan_%s_top8.json" % arm, "w") as fh:
        json.dump({"suffix": "_%s_q" % arm, "layers": layers, "head": False},
                  fh, ensure_ascii=False, indent=2)
print("计划文件已写出：plan_b0_top8.json / plan_b1_top8.json")
PYEOF

# 3) 编译两条新臂（各 8 层，单 worker 串行；不建 head）
echo "--- [2/4] 编译 B0 臂 $(date '+%T')"
CALIB="$MODEL_DIR/calib_b0" SUFFIX=_b0_q LOG_PREFIX=b0 BUILD_HEAD=0 \
    bash build_p1_layers.sh $LAYERS
echo "--- [3/4] 编译 B1 臂 $(date '+%T')"
CALIB="$MODEL_DIR/calib_b1" SUFFIX=_b1_q LOG_PREFIX=b1 BUILD_HEAD=0 \
    bash build_p1_layers.sh $LAYERS

# 4) 评估：四条臂在 pilot / p1test 上的连续轨迹指标
echo "--- [4/4] 评估 $(date '+%T')"
for txt in pilot p1test; do
  for arm in b0 b1 p1; do
    "$PY" ab_run_arm.py --plan-json "$(cat plan_${arm}_top8.json)" --tokens 64 \
        --text "$txt" --out "arm_${arm}_${txt}.npz" --meta "meta_${arm}_${txt}.json"
  done
  "$PY" ab_compare.py --ref "arm_fp16_${txt}.npz" \
      --arm "arm_b0_${txt}.npz" --arm "arm_b1_${txt}.npz" --arm "arm_p1_${txt}.npz" \
      --meta "meta_b0_${txt}.json" --meta "meta_b1_${txt}.json" --meta "meta_p1_${txt}.json" \
      --out "p2b_compare_${txt}.json"
done

echo "--- 收尾 $(date '+%F %T')"
free -m | head -2
ls -l p2b_compare_*.json 2>/dev/null
rm -f "$MODEL_DIR/experiment.lock"
pkill -f experiment_guard.sh 2>/dev/null
trap - EXIT
sleep 2
bash start_service.sh 8000
echo "########## P2-B 完成（网页服务已恢复）$(date '+%F %T') ##########"
