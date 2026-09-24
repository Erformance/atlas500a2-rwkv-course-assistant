#!/bin/bash
# P1b：用新校准（_p1_q）重测逐层敏感度 → 重算帕累托 → 落计划文件 → 文本抽检 → 关机。
#
# 为什么要单独跑：P1 已经证明扩展校准能显著降低 KL / 提高 top-1，那么"哪些层可以压"
# 的排序也应该用新数据重新测一遍，才能把档位重建在新校准上（旧排序来自 _cp_q 的测量）。
#
# 日志：p1b_chain.log；逐层结果 sensitivity_p1.json（可中断续跑），帕累托 pareto_p1.json。
# 说明：期间独占 NPU，网页服务停止；结束用 poweroff -f（上次普通 poweroff 被 60s 硬件
# 看门狗打断成了重启，这次先把现场落盘再强制断电，缩短关机窗口）。
#
# 用法：setsid nohup bash p1b_chain.sh < /dev/null > /dev/null 2>&1 &

MODEL_DIR=/home/disk/models/rwkv7-2.9b
LOG=$MODEL_DIR/p1b_chain.log
PY=/home/disk/miniconda3/envs/npu22/bin/python

cd "$MODEL_DIR" || exit 1
exec >> "$LOG" 2>&1
source /home/disk/cann80base/ascend-toolkit/set_env.sh

echo
echo "########## P1b 开始 $(date '+%F %T') ##########"

bash start_service.sh stop
touch "$MODEL_DIR/experiment.lock"
pgrep -f experiment_guard.sh > /dev/null || \
    setsid nohup bash experiment_guard.sh > /dev/null 2>&1 &
trap 'rm -f "$MODEL_DIR/experiment.lock"; pkill -f experiment_guard.sh' EXIT
free -m | head -2
npu-smi info 2>&1 | sed -n '7p'

# 1) 逐层敏感度：33 个目标各自独立进程（引擎只加载一份，避免设备内存累积）
echo "--- [1/4] 逐层敏感度（_p1_q）$(date '+%T')"
"$PY" sensitivity.py --suffix _p1_q --out sensitivity_p1.json \
    || { echo "敏感度测量失败"; exit 1; }

# 2) 帕累托：按 KL 从小到大取前 N 个单元做 int8
echo "--- [2/4] 帕累托曲线 $(date '+%T')"
"$PY" pareto.py --suffix _p1_q --sens sensitivity_p1.json --out pareto_p1.json \
    --sizes 8,16,24,33 || { echo "帕累托失败"; exit 1; }

# 3) 落计划文件
echo "--- [3/4] 生成计划文件 $(date '+%T')"
"$PY" make_plans_p1.py --pareto pareto_p1.json --prefix plan_p1_top

# 4) 文本抽检：新计划下的自然问答 + 代码题（与 _cp_q 的同题输出可直接对照）
echo "--- [4/4] 文本抽检 $(date '+%T')"
for plan in plan_p1_top8.json plan_p1_top16.json; do
  [ -f "$plan" ] || continue
  echo "======== $plan ========"
  printf '%s\n' "什么是熵？" "/reset" "写一个判断素数的Python函数" "/quit" \
    | "$PY" rwkv7_chat.py --serve --tokens 160 --plan-file "$plan" \
      2>&1 | grep --line-buffered -vE '^\[(INFO|EVENT|WARNING)\]'
done

echo "--- 收尾 $(date '+%F %T') ---"
free -m | head -2
npu-smi info 2>&1 | sed -n '7p'
ls -l sensitivity_p1.json pareto_p1.json plan_p1_top*.json 2>/dev/null

rm -f "$MODEL_DIR/experiment.lock"
pkill -f experiment_guard.sh 2>/dev/null
trap - EXIT
sync
sleep 3
echo "########## 强制断电 $(date '+%F %T') ##########"
sync
poweroff -f
