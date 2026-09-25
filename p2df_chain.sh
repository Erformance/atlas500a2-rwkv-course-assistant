#!/bin/bash
# P2-D + P2-F 补充实验链：跑完恢复网页服务（白天不关机）。
#   P2-D：单次量化脉冲（无传播）vs 持续量化（有传播），看便宜代理能否预测持续结果
#   P2-F：强制 token vs 自由生成，看输出决策反馈把偏差放大了多少
#
# 日志：p2df_chain.log；产物 p2d_result.json / p2f_result.json
# 注意：/tmp 是 5.7G 内存盘（算在 11.5G 共享内存里），参考轨迹 683MB 一律放磁盘目录。
#
# 用法：setsid nohup bash p2df_chain.sh < /dev/null > /dev/null 2>&1 &

MODEL_DIR=/home/disk/models/rwkv7-2.9b
LOG=$MODEL_DIR/p2df_chain.log
PY=/home/disk/miniconda3/envs/npu22/bin/python
REF=$MODEL_DIR/p2c_ref

cd "$MODEL_DIR" || exit 1
exec >> "$LOG" 2>&1
source /home/disk/cann80base/ascend-toolkit/set_env.sh

echo
echo "########## P2-D/F 开始 $(date '+%F %T') ##########"

bash start_service.sh stop
touch "$MODEL_DIR/experiment.lock"
pgrep -f experiment_guard.sh > /dev/null || \
    setsid nohup bash experiment_guard.sh > /dev/null 2>&1 &
trap 'rm -f "$MODEL_DIR/experiment.lock"; pkill -f experiment_guard.sh' EXIT
free -m | head -2

# 0) fp16 参考轨迹（32 步：每步状态 + logits）。上次在 /tmp 里，重启后没了，重算。
if [ ! -f "$REF/logits.npy" ]; then
  echo "--- [0/3] 生成 fp16 参考轨迹 $(date '+%T')"
  "$PY" p2c_ref_states.py --tokens 32 --out-dir "$REF" || { echo "参考轨迹失败"; exit 1; }
fi

# 1) P2-D
echo "--- [1/3] P2-D 脉冲 vs 持续 $(date '+%T')"
"$PY" p2d_pulse.py --ref-dir "$REF" --out "$MODEL_DIR/p2d_result.json" \
    || echo "P2-D 失败（见日志）"

# 2) P2-F
echo "--- [2/3] P2-F 强制 vs 自由生成 $(date '+%T')"
"$PY" p2f_free_gen.py --prefix 64 --tokens 64 --out "$MODEL_DIR/p2f_result.json" \
    || echo "P2-F 失败（见日志）"

# 3) 回填 manifest（协议 §1：数据分割 ID/哈希、候选计划、候选 OM 哈希）
echo "--- [3/4] 回填 manifest $(date '+%T')"
"$PY" update_manifest_p1.py || echo "manifest 回填失败（见日志）"

# 4) 收尾：释放锁，恢复网页服务
echo "--- [4/4] 收尾 $(date '+%F %T')"
free -m | head -2
ls -l p2d_result.json p2f_result.json 2>/dev/null
rm -f "$MODEL_DIR/experiment.lock"
pkill -f experiment_guard.sh 2>/dev/null
trap - EXIT
sleep 2
bash start_service.sh 8000
echo "########## P2-D/F 完成 $(date '+%F %T') ##########"
