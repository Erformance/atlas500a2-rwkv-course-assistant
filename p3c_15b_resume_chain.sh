#!/bin/bash
# P3C 续跑（1.5B 量化迁移）：昨晚的链在 pack 步骤因 pack_calib_p1.py 写死 32 层而死掉。
# 现场：calib_m（多年龄，24 层已打包）齐全；calib_a0 未采。本链从现场继续：
#   1) 采/打包 a0 校准（32 段 × 年龄 0，同样是 32 组样本）
#   2) 用多年龄校准编译 24 层
#   3) fp16 参考臂 + 24 目标敏感度扫描 → 取 top8 计划 → 实测
#   4) 同 8 层用 a0 校准再编译一遍 → 实测同一计划 → 对比
#   5) 收尾：强制断电（三次重试，避免像昨晚那样没断成）
# 日志：p3c_15b_resume.log
#
# 用法：setsid nohup bash p3c_15b_resume_chain.sh < /dev/null > /dev/null 2>&1 &

MD=/home/disk/models/rwkv7-1.5b
SRC=/home/disk/models/rwkv7-2.9b
LOG=$MD/p3c_15b_resume.log
PY=/home/disk/miniconda3/envs/npu22/bin/python
SUF_M=_15b_m_q
SUF_A=_15b_a0_q
export RWKV_MODEL_DIR=$MD

cd "$MD" || exit 1
exec >> "$LOG" 2>&1
source /home/disk/cann80base/ascend-toolkit/set_env.sh

echo
echo "########## P3C 续跑开始 $(date '+%F %T') ##########"

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

# ---- 1) a0 校准（32 段 × 年龄 0）-------------------------------------------
if [ ! -f "$MD/calib_a0/layer00.npz" ]; then
  echo "--- [1/5] 采集/打包 a0 校准 $(date '+%T')"
  "$PY" "$SRC/capture_calib_p1.py" --split calib --limit 32 --ages 0 \
      --out "$MD/calib_a0_raw" --max-samples 32 --manifest "$MD/manifest_15b_a0.json" || exit 1
  "$PY" "$SRC/pack_calib_p1.py" --raw "$MD/calib_a0_raw" --out "$MD/calib_a0" || exit 1
fi
echo "  calib_m 层数 $(ls $MD/calib_m/*.npz | wc -l)｜calib_a0 层数 $(ls $MD/calib_a0/*.npz | wc -l)"

# ---- 2) 编译 24 层（多年龄校准）-------------------------------------------
echo "--- [2/5] 编译 24 层（多年龄校准）$(date '+%T')"
SCRIPT_DIR="$SRC" CALIB="$MD/calib_m" SUFFIX=$SUF_M LOG_PREFIX=15b_m BUILD_HEAD=0 MODEL_DIR="$MD" \
    bash "$SRC/build_p1_layers.sh" $(seq 0 23)
echo "  已编译：$(ls $MD/layer*${SUF_M}.om 2>/dev/null | wc -l)/24"

# ---- 3) fp16 参考臂 + 敏感度扫描 ------------------------------------------
echo "--- [3/5] fp16 参考臂 + 24 目标敏感度 $(date '+%T')"
"$PY" "$SRC/ab_run_arm.py" --suffix "" --tokens 64 --text pilot \
    --out "$MD/arm_fp16_15b.npz" --meta "$MD/meta_fp16_15b.json" || echo "参考臂失败"
"$PY" "$SRC/sensitivity.py" --suffix $SUF_M --tokens 64 --ref "$MD/arm_fp16_15b.npz" \
    --targets "$(seq -s, 0 23)" --out "$MD/sensitivity_15b.json" || echo "敏感度有失败项"

# ---- 4) top8 计划 + 两条臂实测 --------------------------------------------
echo "--- [4/5] top8 计划与两条臂实测 $(date '+%T')"
"$PY" - "$MD" $SUF_M $SUF_A <<'PYEOF'
import json, os, sys
md, suf_m, suf_a = sys.argv[1], sys.argv[2], sys.argv[3]
p = os.path.join(md, "sensitivity_15b.json")
tg = json.load(open(p))["targets"]
ranked = sorted(tg.items(), key=lambda kv: kv[1]["kl_mean"])
safe = [n for n, _ in ranked if n != "head"][:8]
for suf, name in ((suf_m, "plan_15b_m_top8.json"), (suf_a, "plan_15b_a0_top8.json")):
    json.dump({"suffix": suf, "layers": [int(x) for x in safe], "head": False},
              open(os.path.join(md, name), "w"), ensure_ascii=False, indent=2)
print("1.5B 最安全 8 层：", safe)
print("最敏感 5 个：", [(n, round(v["kl_mean"], 4)) for n, v in ranked[-5:]])
PYEOF
PLAN8=$("$PY" -c "import json;print(' '.join(str(x) for x in json.load(open('$MD/plan_15b_m_top8.json'))['layers']))")
SCRIPT_DIR="$SRC" CALIB="$MD/calib_a0" SUFFIX=$SUF_A LOG_PREFIX=15b_a0 BUILD_HEAD=0 MODEL_DIR="$MD" \
    bash "$SRC/build_p1_layers.sh" $PLAN8
"$PY" "$SRC/ab_run_arm.py" --plan-json "$(cat $MD/plan_15b_m_top8.json)" --tokens 64 \
    --text pilot --out "$MD/arm_15b_m.npz" --meta "$MD/meta_15b_m.json" || echo "多年龄臂失败"
"$PY" "$SRC/ab_run_arm.py" --plan-json "$(cat $MD/plan_15b_a0_top8.json)" --tokens 64 \
    --text pilot --out "$MD/arm_15b_a0.npz" --meta "$MD/meta_15b_a0.json" || echo "a0 臂失败"
"$PY" "$SRC/ab_compare.py" --ref "$MD/arm_fp16_15b.npz" \
    --arm "$MD/arm_15b_m.npz" --arm "$MD/arm_15b_a0.npz" \
    --meta "$MD/meta_15b_m.json" --meta "$MD/meta_15b_a0.json" \
    --out "$MD/p3c_15b_compare.json" || echo "比对失败"

# ---- 5) 收尾 + 强制断电（重试三次）----------------------------------------
echo "--- [5/5] 收尾 $(date '+%F %T')"
ls -l "$MD/sensitivity_15b.json" "$MD/p3c_15b_compare.json" 2>/dev/null
free -m | head -2
npu-smi info 2>&1 | sed -n '7p'
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
echo "########## 断电命令已发 3 次 $(date '+%F %T') ##########"
