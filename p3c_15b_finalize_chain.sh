#!/bin/bash
# P3C 收尾（1.5B 量化迁移）：
#   1) 24 目标敏感度扫描（修好脚本目录后重跑）
#   2) 由扫描取 top8 → 用多年龄校准实测该计划
#   3) 确保 a0 臂有这 8 层（已有的跳过）→ 实测同一计划
#   4) 对比 → 强制断电（三重试）
# 日志：p3c_15b_finalize.log
MD=/home/disk/models/rwkv7-1.5b
SRC=/home/disk/models/rwkv7-2.9b
LOG=$MD/p3c_15b_finalize.log
PY=/home/disk/miniconda3/envs/npu22/bin/python
export RWKV_MODEL_DIR=$MD RWKV_SCRIPT_DIR=$SRC
SUF_M=_15b_m_q
SUF_A=_15b_a0_q

cd "$MD" || exit 1
exec >> "$LOG" 2>&1
source /home/disk/cann80base/ascend-toolkit/set_env.sh

echo
echo "########## P3C 收尾开始 $(date '+%F %T') ##########"
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

echo "--- [1/4] 24 目标敏感度扫描 $(date '+%T')"
"$PY" "$SRC/sensitivity.py" --suffix $SUF_M --tokens 64 --ref "$MD/arm_fp16_15b.npz" \
    --targets "$(seq -s, 0 23)" --out "$MD/sensitivity_15b.json" || echo "敏感度有失败项"

echo "--- [2/4] top8 计划 + 多年龄臂实测 $(date '+%T')"
"$PY" - "$MD" $SUF_M $SUF_A <<'PYEOF'
import json, os, sys
md, suf_m, suf_a = sys.argv[1], sys.argv[2], sys.argv[3]
tg = json.load(open(os.path.join(md, "sensitivity_15b.json")))["targets"]
ranked = sorted(tg.items(), key=lambda kv: kv[1]["kl_mean"])
safe = [n for n, _ in ranked if n != "head"][:8]
for suf, name in ((suf_m, "plan_15b_m_top8.json"), (suf_a, "plan_15b_a0_top8.json")):
    json.dump({"suffix": suf, "layers": [int(x) for x in safe], "head": False},
              open(os.path.join(md, name), "w"), ensure_ascii=False, indent=2)
print("1.5B 最安全 8 层：", safe)
print("最敏感 6 个：", [(n, round(v["kl_mean"], 4)) for n, v in ranked[-6:]])
PYEOF
PLAN8=$("$PY" -c "import json;print(' '.join(str(x) for x in json.load(open('$MD/plan_15b_m_top8.json'))['layers']))")
echo "  计划层：$PLAN8"
"$PY" "$SRC/ab_run_arm.py" --plan-json "$(cat $MD/plan_15b_m_top8.json)" --tokens 64 \
    --text pilot --out "$MD/arm_15b_m.npz" --meta "$MD/meta_15b_m.json" || echo "多年龄臂失败"

echo "--- [3/4] a0 臂补齐这 8 层并实测 $(date '+%T')"
SCRIPT_DIR="$SRC" CALIB="$MD/calib_a0" SUFFIX=$SUF_A LOG_PREFIX=15b_a0 BUILD_HEAD=0 MODEL_DIR="$MD" \
    bash "$SRC/build_p1_layers.sh" $PLAN8
"$PY" "$SRC/ab_run_arm.py" --plan-json "$(cat $MD/plan_15b_a0_top8.json)" --tokens 64 \
    --text pilot --out "$MD/arm_15b_a0.npz" --meta "$MD/meta_15b_a0.json" || echo "a0 臂失败"

echo "--- [4/4] 对比 + 收尾 $(date '+%T')"
"$PY" "$SRC/ab_compare.py" --ref "$MD/arm_fp16_15b.npz" \
    --arm "$MD/arm_15b_m.npz" --arm "$MD/arm_15b_a0.npz" \
    --meta "$MD/meta_15b_m.json" --meta "$MD/meta_15b_a0.json" \
    --out "$MD/p3c_15b_compare.json" || echo "比对失败"
ls -l "$MD/sensitivity_15b.json" "$MD/plan_15b_m_top8.json" "$MD/p3c_15b_compare.json" 2>/dev/null
free -m | head -2
rm -f "$SRC/experiment.lock"
pkill -f experiment_guard.sh 2>/dev/null
trap - EXIT
sync
sleep 5
for i in 1 2 3; do
  echo "[$(date '+%F %T')] 第 $i 次强制断电"
  sync
  poweroff -f
  sleep 20
done
