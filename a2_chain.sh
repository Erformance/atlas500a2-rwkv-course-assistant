#!/bin/bash
# 任务书 Task A（公平 calibration baseline）完整链：
#   A1 = 现有 _b0_q（32 段 × 年龄 0）
#   A2 = 新采 _a2_q（16 段 × 2 个随机位置，seed 固定）      ← 本链新增的唯一一条臂
#   A3 = 现有 _b1_q（16 段 × {0,256}）
#   A4 = 现有 _p1_q（32 段 × {0,64,256}，高预算参照）
# 四臂同一模型/同一量化工具链/同一 8 层计划（24,6,11,22,23,20,25,28，head=False）。
# 评估口径：pilot 与 p1test 两份文本，64 步（与既有 P2-B 可比）与 128 步（任务书要求的长窗口）。
# 日志：a2_chain.log；结果：final_calibration_baselines.json、p1_a2_capture_manifest.json
#
# 用法：setsid nohup bash a2_chain.sh < /dev/null > /dev/null 2>&1 &

MODEL_DIR=/home/disk/models/rwkv7-2.9b
LOG=$MODEL_DIR/a2_chain.log
PY=/home/disk/miniconda3/envs/npu22/bin/python
LAYERS="24 6 11 22 23 20 25 28"
SEED=${SEED:-20261003}

cd "$MODEL_DIR" || exit 1
exec >> "$LOG" 2>&1
source /home/disk/cann80base/ascend-toolkit/set_env.sh

echo
echo "########## Task A（A2 Random Trajectory）开始 $(date '+%F %T') ##########"

for p in $(pgrep -f "bash $(basename "$0")"); do
  [ "$p" != "$$" ] && { echo "已有另一份在跑（pid $p），退出"; exit 1; }
done

# 等别的实验链释放锁（等不到就退出，绝不并发——9-28 的教训）
for i in $(seq 1 480); do
  [ -f "$MODEL_DIR/experiment.lock" ] || break
  echo "  等待别的实验链释放 experiment.lock（$i）"
  sleep 60
done
[ -f "$MODEL_DIR/experiment.lock" ] && { echo "锁仍被占用，退出"; exit 1; }

bash start_service.sh stop
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

T0=$(date +%s)

# 1) 采 A2 校准数据（32 组随机位置）
if [ ! -f "$MODEL_DIR/calib_a2/layer00.npz" ]; then
  echo "--- [1/5] 采集 A2（16 段 × 2 个随机位置，seed=$SEED）$(date '+%T')"
  "$PY" capture_calib_a2.py --split calib --limit 16 --per-seg 2 \
      --seed "$SEED" --traj-max 512 \
      --out "$MODEL_DIR/calib_a2_raw" \
      --manifest "$MODEL_DIR/p1_a2_capture_manifest.json" || exit 1
  "$PY" pack_calib_p1.py --raw "$MODEL_DIR/calib_a2_raw" --out "$MODEL_DIR/calib_a2" || exit 1
else
  echo "--- [1/5] A2 校准数据已存在，跳过"
fi
T_CAP=$(date +%s)

# 2) 四条臂的计划文件（同一组层，只有 suffix 不同）
"$PY" - <<'PYEOF'
import json
layers = [24, 6, 11, 22, 23, 20, 25, 28]
for arm, suf in (("a1", "_b0_q"), ("a2", "_a2_q"), ("a3", "_b1_q"), ("a4", "_p1_q")):
    with open("plan_%s_top8.json" % arm, "w") as fh:
        json.dump({"suffix": suf, "layers": layers, "head": False}, fh,
                  ensure_ascii=False, indent=2)
print("计划文件已写出：plan_a1..a4_top8.json")
PYEOF

# 3) 只编译新增的 A2 臂（8 层，单 worker 串行；不建 head）
echo "--- [2/5] 编译 A2 臂 $(date '+%T')"
CALIB="$MODEL_DIR/calib_a2" SUFFIX=_a2_q LOG_PREFIX=a2 BUILD_HEAD=0 \
    bash build_p1_layers.sh $LAYERS
T_QUANT=$(date +%s)

# 4) 评估：四臂 × 两份文本 × 64/128 步
echo "--- [3/5] 评估 64 步 $(date '+%T')"
for txt in pilot p1test; do
  for arm in a1 a2 a3 a4; do
    [ -f "arm_a_${arm}_${txt}_64.npz" ] && continue
    "$PY" ab_run_arm.py --plan-json "$(cat plan_${arm}_top8.json)" --tokens 64 \
        --text "$txt" --out "arm_a_${arm}_${txt}_64.npz" \
        --meta "meta_a_${arm}_${txt}_64.json" --decode-steps 8
  done
  "$PY" ab_compare.py --ref "arm_fp16_${txt}.npz" \
      --arm "arm_a_a1_${txt}_64.npz" --arm "arm_a_a2_${txt}_64.npz" \
      --arm "arm_a_a3_${txt}_64.npz" --arm "arm_a_a4_${txt}_64.npz" \
      --meta "meta_a_a1_${txt}_64.json" --meta "meta_a_a2_${txt}_64.json" \
      --meta "meta_a_a3_${txt}_64.json" --meta "meta_a_a4_${txt}_64.json" \
      --out "final_a_compare_${txt}_64.json"
done

echo "--- [4/5] 评估 128 步 $(date '+%T')"
for txt in pilot p1test; do
  for arm in a1 a2 a3 a4; do
    [ -f "arm_a_${arm}_${txt}_128.npz" ] && continue
    "$PY" ab_run_arm.py --plan-json "$(cat plan_${arm}_top8.json)" --tokens 128 \
        --text "$txt" --out "arm_a_${arm}_${txt}_128.npz" \
        --meta "meta_a_${arm}_${txt}_128.json" --decode-steps 8
  done
  "$PY" ab_compare.py --ref "arm_fp16_128_${txt}.npz" \
      --arm "arm_a_a1_${txt}_128.npz" --arm "arm_a_a2_${txt}_128.npz" \
      --arm "arm_a_a3_${txt}_128.npz" --arm "arm_a_a4_${txt}_128.npz" \
      --meta "meta_a_a1_${txt}_128.json" --meta "meta_a_a2_${txt}_128.json" \
      --meta "meta_a_a3_${txt}_128.json" --meta "meta_a_a4_${txt}_128.json" \
      --out "final_a_compare_${txt}_128.json"
done
T_EVAL=$(date +%s)

# 5) 汇总成任务书要求的 final_calibration_baselines.json
echo "--- [5/5] 汇总 $(date '+%T')"
"$PY" final_a_summary.py --seed "$SEED" \
    --t-cap "$T_CAP" --t-quant "$T_QUANT" --t-eval "$T_EVAL" --t0 "$T0" || echo "汇总失败"

echo "--- 收尾 $(date '+%F %T')"
free -m | head -2
ls -l final_calibration_baselines.json final_a_compare_*_*.json 2>/dev/null
rm -f "$MODEL_DIR/experiment.lock"
pkill -f experiment_guard.sh 2>/dev/null
trap - EXIT
sleep 2
bash start_service.sh 8000
echo "########## Task A 完成（网页服务已恢复）$(date '+%F %T') ##########"
