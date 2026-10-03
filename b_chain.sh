#!/bin/bash
# 任务书 Task B：等预算 mixed-precision 选层（固定 K=8）的评估链。
#   * 计划由 final_select_k8.py 从已有 profiling 产物生成（含 20 个随机 seed）；
#   * 全部计划共用线上 _p1_q 校准（P3 阶段已把 33 个目标全编译），**零编译成本**；
#   * 每个计划在 pilot（必做）与 p1test（时间允许）上跑 128 步，和 fp16 参考比 KL / top-1 / tok/s；
#   * 汇总成任务书 §3.5 的表格 → final_selection_k8.json。
# 日志：b_chain.log
#
# 用法：setsid nohup bash b_chain.sh < /dev/null > /dev/null 2>&1 &

MODEL_DIR=/home/disk/models/rwkv7-2.9b
LOG=$MODEL_DIR/b_chain.log
PY=/home/disk/miniconda3/envs/npu22/bin/python
TEXTS=${TEXTS:-"pilot p1test"}

cd "$MODEL_DIR" || exit 1
exec >> "$LOG" 2>&1
source /home/disk/cann80base/ascend-toolkit/set_env.sh

echo
echo "########## Task B（K=8 选层）开始 $(date '+%F %T') ##########"

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

# 1) 生成计划文件
echo "--- [1/3] 生成 K=8 计划 $(date '+%T')"
"$PY" final_select_k8.py --write-plans || exit 1
ls -1 plan_sel_*.json | wc -l

# 2) 逐计划评估（128 步）
echo "--- [2/3] 评估各计划（128 步）$(date '+%T')"
for txt in $TEXTS; do
  for f in plan_sel_*.json; do
    name=$(basename "$f" .json | sed 's/^plan_sel_//')
    out="arm_sel_${name}_${txt}_128.npz"
    [ -f "$out" ] && continue
    echo "  [$txt] $name"
    "$PY" ab_run_arm.py --plan-json "$(cat "$f")" --tokens 128 --text "$txt" \
        --out "$out" --meta "meta_sel_${name}_${txt}_128.json" --decode-steps 8 \
        || echo "    臂 $name 失败（保留现场，继续）"
  done
done

# 3) 汇总
echo "--- [3/3] 汇总 $(date '+%T')"
"$PY" final_b_summary.py --texts "$TEXTS" || echo "汇总失败"

echo "--- 收尾 $(date '+%F %T')"
free -m | head -2
ls -l final_selection_k8.json 2>/dev/null
rm -f "$MODEL_DIR/experiment.lock"
pkill -f experiment_guard.sh 2>/dev/null
trap - EXIT
sleep 2
bash start_service.sh 8000
echo "########## Task B 完成（网页服务已恢复）$(date '+%F %T') ##########"
