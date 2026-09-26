#!/bin/bash
# P3 方向指标全链（协议 §5 的"输出加权 Gramian"一阶形式）：
#   1) NPU 上实测每层误差向量 δᵢ（int8 层 vs fp16 层，同一份真实校准输入）
#   2) CPU（PyTorch）上算 ∂(−log p)/∂(层输出) gᵢ，做 γᵢ = |gᵢ·δᵢ|
#   3) 与 rollout KL / 局部误差比排序预测力，并生成"按 γ 选前 8 层"的计划文件
#   4) 实测该计划的 KL（层都已编译，零编译成本）
# 跑完恢复网页服务（不关机；要关机由人决定）。
# 日志：p3b_dir.log
#
# 用法：setsid nohup bash p3b_dir_chain.sh < /dev/null > /dev/null 2>&1 &

MODEL_DIR=/home/disk/models/rwkv7-2.9b
LOG=$MODEL_DIR/p3b_dir.log
PY_NPU=/home/disk/miniconda3/envs/npu22/bin/python
PY_CPU=/home/disk/miniconda3/envs/rwkv7/bin/python
TOKENS=${TOKENS:-4}

cd "$MODEL_DIR" || exit 1
exec >> "$LOG" 2>&1
source /home/disk/cann80base/ascend-toolkit/set_env.sh

echo
echo "########## P3 方向指标开始 $(date '+%F %T') ##########"

for p in $(pgrep -f "bash $(basename "$0")"); do
  [ "$p" != "$$" ] && { echo "已有另一份在跑，退出"; exit 1; }
done

bash start_service.sh stop
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

# 1) 误差向量（NPU，每层一个子进程）
echo "--- [1/4] 实测每层误差向量 δᵢ $(date '+%T')"
"$PY_NPU" p3b_err_vectors.py --suffix _p1_q --calib calib_p1 --steps 0,1,2 \
    || echo "误差向量采集有失败项"

# 2) 方向灵敏度（CPU + autograd；把 NPU 的东西放掉再加载 PyTorch 权重）
echo "--- [2/4] 方向灵敏度 γᵢ = |gᵢ·δᵢ| $(date '+%T')"
free -m | head -2
"$PY_CPU" p3b_dir_metric.py --tokens "$TOKENS" --text val --out p3b_dir.json \
    || echo "方向灵敏度失败"

# 3) 比对
echo "--- [3/4] 与 rollout KL / 局部误差比对 $(date '+%T')"
"$PY_NPU" p3b_compare.py || echo "比对失败"

# 4) 计划级实测（零编译成本）
echo "--- [4/4] 按方向指标选前 8 层的计划实测 $(date '+%T')"
for i in $(seq 1 24); do
  avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  [ "$avail" -gt 6000 ] && break
  sleep 5
done
if [ -f plan_dir_top8.json ]; then
  "$PY_NPU" ab_run_arm.py --plan-json "$(cat plan_dir_top8.json)" --tokens 64 \
      --text pilot --out arm_dir_pilot.npz --meta meta_dir_pilot.json || echo "计划实测失败"
  "$PY_NPU" ab_compare.py --ref arm_fp16.npz --arm arm_dir_pilot.npz \
      --meta meta_dir_pilot.json --out p3b_plan_compare.json || echo "计划比对失败"
fi

echo "--- 收尾 $(date '+%F %T')"
free -m | head -2
ls -l p3b_dir.json p3b_compare.json p3b_plan_compare.json 2>/dev/null
rm -f "$MODEL_DIR/experiment.lock"
pkill -f experiment_guard.sh 2>/dev/null
trap - EXIT
sleep 2
bash start_service.sh 8000
echo "########## P3 方向指标完成（网页服务已恢复）$(date '+%F %T') ##########"
