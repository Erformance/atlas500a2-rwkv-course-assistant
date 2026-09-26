#!/bin/bash
# P3C（过夜）：1.5B 上的量化迁移实验——检验 2.9B 的两条结论是否跨规模成立：
#   (a) "状态年龄覆盖是主因"：同样 32 组样本，多年龄 vs 仅年龄 0；
#   (b) 选层排序是否与 2.9B 一致（逐层敏感度）。
#
# 流程：两组校准（各 32 组）→ 用多年龄校准编译 24 层 → fp16 参考臂 → 24 目标敏感度扫描
#       → 取该扫描的 top8 作为计划 → 用同一 8 层、仅年龄 0 的校准再编译一遍 → 实测对比
#       → 同步 → 强制断电。
# 日志：p3c_15b.log；产物都在 /home/disk/models/rwkv7-1.5b/
#
# 用法：setsid nohup bash p3c_15b_quant_chain.sh < /dev/null > /dev/null 2>&1 &

MD=/home/disk/models/rwkv7-1.5b
SRC=/home/disk/models/rwkv7-2.9b
LOG=$MD/p3c_15b.log
PY_NPU=/home/disk/miniconda3/envs/npu22/bin/python
SUF_M=_15b_m_q          # 多年龄校准
SUF_A=_15b_a0_q         # 仅年龄 0 校准
export RWKV_MODEL_DIR=$MD

cd "$MD" || exit 1
exec >> "$LOG" 2>&1
source /home/disk/cann80base/ascend-toolkit/set_env.sh
export HF_HOME=/home/disk/hf_cache HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_TELEMETRY=1

echo
echo "########## P3C 1.5B 量化迁移开始 $(date '+%F %T') ##########"

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

# ---- 1) 两组校准数据（各 32 组样本）----------------------------------------
if [ ! -f "$MD/calib_m/layer00.npz" ]; then
  echo "--- [1/6] 多年龄校准（16 段 × 年龄 0/256）$(date '+%T')"
  "$PY_NPU" "$SRC/capture_calib_p1.py" --split calib --limit 16 --ages 0,256 \
      --out "$MD/calib_m_raw" --max-samples 32 --manifest "$MD/manifest_15b_m.json" || exit 1
  "$PY_NPU" "$SRC/pack_calib_p1.py" --raw "$MD/calib_m_raw" --out "$MD/calib_m" || exit 1
fi
if [ ! -f "$MD/calib_a0/layer00.npz" ]; then
  echo "--- [1/6] 仅年龄 0 校准（32 段 × 年龄 0）$(date '+%T')"
  "$PY_NPU" "$SRC/capture_calib_p1.py" --split calib --limit 32 --ages 0 \
      --out "$MD/calib_a0_raw" --max-samples 32 --manifest "$MD/manifest_15b_a0.json" || exit 1
  "$PY_NPU" "$SRC/pack_calib_p1.py" --raw "$MD/calib_a0_raw" --out "$MD/calib_a0" || exit 1
fi

# ---- 2) 用多年龄校准编译全部 24 层 ----------------------------------------
echo "--- [2/6] 编译 24 层（多年龄校准）$(date '+%T')"
CALIB="$MD/calib_m" SUFFIX=$SUF_M LOG_PREFIX=15b_m BUILD_HEAD=0 MODEL_DIR="$MD" \
    bash "$SRC/build_p1_layers.sh" $(seq 0 23)
echo "  已编译：$(ls $MD/layer*${SUF_M}.om 2>/dev/null | wc -l)/24"

# ---- 3) fp16 参考臂 + 24 目标敏感度扫描 -----------------------------------
echo "--- [3/6] fp16 参考臂 $(date '+%T')"
"$PY_NPU" "$SRC/ab_run_arm.py" --suffix "" --tokens 64 --text pilot \
    --out "$MD/arm_fp16_15b.npz" --meta "$MD/meta_fp16_15b.json" || echo "参考臂失败"
echo "--- [4/6] 24 目标敏感度扫描 $(date '+%T')"
"$PY_NPU" "$SRC/sensitivity.py" --suffix $SUF_M --tokens 64 --ref "$MD/arm_fp16_15b.npz" \
    --out "$MD/sensitivity_15b.json" || echo "敏感度扫描有失败项"

# ---- 4) 由敏感度排序取 top8 计划 ------------------------------------------
echo "--- [5/6] 生成 1.5B top8 计划 $(date '+%T')"
"$PY_NPU" - "$MD" $SUF_M <<'PYEOF'
import json, os, sys
md, suf = sys.argv[1], sys.argv[2]
p = os.path.join(md, "sensitivity_15b.json")
if not os.path.exists(p):
    print("没有 sensitivity_15b.json，跳过"); raise SystemExit(0)
tg = json.load(open(p))["targets"]
ranked = sorted(tg.items(), key=lambda kv: kv[1]["kl_mean"])
safe = [n for n, _ in ranked if n != "head"][:8]
plan = {"suffix": suf, "layers": [int(x) for x in safe], "head": False}
json.dump(plan, open(os.path.join(md, "plan_15b_m_top8.json"), "w"), ensure_ascii=False, indent=2)
print("1.5B 最安全的 8 层（多年龄校准）：", plan["layers"])
for n, v in ranked[:10]:
    print("   %-4s KL %.4f top-1 %.1f%%" % (n, v["kl_mean"], v["top1_agreement"] * 100))
PYEOF

# ---- 5) 计划实测：多年龄 vs 仅年龄 0（同 8 层）----------------------------
PLAN8=$("$PY_NPU" -c "import json;print(' '.join(str(x) for x in json.load(open('$MD/plan_15b_m_top8.json'))['layers']))")
echo "--- [6/6] 同 8 层两种校准的实测对比（层：$PLAN8）$(date '+%T')"
if [ -z "$PLAN8" ]; then
  echo "计划为空，跳过实测"
else
  CALIB="$MD/calib_a0" SUFFIX=$SUF_A LOG_PREFIX=15b_a0 BUILD_HEAD=0 MODEL_DIR="$MD" \
      bash "$SRC/build_p1_layers.sh" $PLAN8
  "$PY_NPU" "$SRC/ab_run_arm.py" --plan-json "$(cat $MD/plan_15b_m_top8.json)" \
      --tokens 64 --text pilot --out "$MD/arm_15b_m.npz" --meta "$MD/meta_15b_m.json" || echo "多年龄臂失败"
  "$PY_NPU" - "$MD" $SUF_A <<'PYEOF'
import json, os, sys
md, suf = sys.argv[1], sys.argv[2]
plan = json.load(open(os.path.join(md, "plan_15b_m_top8.json")))
plan["suffix"] = suf
json.dump(plan, open(os.path.join(md, "plan_15b_a0_top8.json"), "w"), ensure_ascii=False, indent=2)
print("已写 plan_15b_a0_top8.json（同 8 层，仅年龄 0 校准）")
PYEOF
  "$PY_NPU" "$SRC/ab_run_arm.py" --plan-json "$(cat $MD/plan_15b_a0_top8.json)" \
      --tokens 64 --text pilot --out "$MD/arm_15b_a0.npz" --meta "$MD/meta_15b_a0.json" || echo "仅年龄0臂失败"
  "$PY_NPU" "$SRC/ab_compare.py" --ref "$MD/arm_fp16_15b.npz" \
      --arm "$MD/arm_15b_m.npz" --arm "$MD/arm_15b_a0.npz" \
      --meta "$MD/meta_15b_m.json" --meta "$MD/meta_15b_a0.json" \
      --out "$MD/p3c_15b_compare.json" || echo "比对失败"
fi

# ---- 收尾 -----------------------------------------------------------------
echo "--- 收尾 $(date '+%F %T')"
ls -l "$MD"/layer*${SUF_M}.om 2>/dev/null | wc -l
ls -l "$MD"/layer*${SUF_A}.om 2>/dev/null | wc -l
ls -l "$MD/sensitivity_15b.json" "$MD/p3c_15b_compare.json" 2>/dev/null
free -m | head -2
rm -f "$SRC/experiment.lock"
pkill -f experiment_guard.sh 2>/dev/null
trap - EXIT
sync
sleep 5
echo "########## P3C 完成，强制断电 $(date '+%F %T') ##########"
sync
poweroff -f
