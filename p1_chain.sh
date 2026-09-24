#!/bin/bash
# P1 全链（协议 §3）：独占 NPU → 扩展校准数据（32 段 × 状态年龄 0/64/256 = 96 组）
# → 打包 → 重新量化并串行编译 32 层 + 输出头（_p1_q）→ 与 fp16 / 旧 int8 对照 →
# 汇总 → 收尾 → 自动关机。
#
# 日志：p1_chain.log（各阶段另有自己的日志）
# 说明：全程持有 experiment.lock，5 分钟健康看门狗不会介入；网页服务在跑本条链期间是停的。
#
# 用法：setsid nohup bash p1_chain.sh < /dev/null > /dev/null 2>&1 &

MODEL_DIR=/home/disk/models/rwkv7-2.9b
LOG=$MODEL_DIR/p1_chain.log
PY=/home/disk/miniconda3/envs/npu22/bin/python

cd "$MODEL_DIR" || exit 1
exec >> "$LOG" 2>&1
# 让所有子进程都能 import acl（pyACL 的 python 包在这个环境里）
source /home/disk/cann80base/ascend-toolkit/set_env.sh

echo
echo "########## P1 开始 $(date '+%F %T') ##########"

# 0) 独占设备：停网页服务 → 上实验锁 → 起内存看门狗
bash start_service.sh stop
touch "$MODEL_DIR/experiment.lock"
pgrep -f experiment_guard.sh > /dev/null || \
    setsid nohup bash experiment_guard.sh > /dev/null 2>&1 &
# 异常退出也要把锁和看门狗收干净，否则下次开机自启会被锁挡住
trap 'rm -f "$MODEL_DIR/experiment.lock"; pkill -f experiment_guard.sh' EXIT
free -m | head -2
npu-smi info 2>&1 | sed -n '7p'

# 1) 语料清单（哈希进 manifest）
echo "--- [1/6] 语料清单 $(date '+%T')"
"$PY" p1_corpus.py --manifest "$MODEL_DIR/p1_corpus_manifest.json" || exit 1
"$PY" p1_corpus.py --list || exit 1

# 2) 抓校准数据（memmap，逐段打印进度）
echo "--- [2/6] 采集校准数据 $(date '+%T')"
"$PY" capture_calib_p1.py --split calib --out "$MODEL_DIR/calib_p1_raw" \
    --ages 0,64,256 --max-samples 96 || { echo "采集失败"; exit 1; }

# 3) 打包成 quantize_layer_calib.py 能吃的 npz
echo "--- [3/6] 打包 npz $(date '+%T')"
"$PY" pack_calib_p1.py --raw "$MODEL_DIR/calib_p1_raw" --out "$MODEL_DIR/calib_p1" || exit 1

# 4) 量化 + 编译（单 worker 串行，约 2~4 分钟/层）
echo "--- [4/6] 量化 + ATC 编译 $(date '+%T')"
bash build_p1_layers.sh || echo "构建阶段有失败项，见 quant_p1_*.log / atc_p1_*.log"

# 5) 评估：fp16 / 旧 int8（_cp_q）/ 新 int8（_p1_q）三条臂，测试分割 + pilot 文本
echo "--- [5/6] 评估 $(date '+%T')"
for txt in p1test pilot; do
  "$PY" ab_run_arm.py --suffix ""      --tokens 64 --text "$txt" \
        --out "arm_fp16_$txt.npz" --meta "meta_fp16_$txt.json"
  "$PY" ab_run_arm.py --suffix "_cp_q" --tokens 64 --text "$txt" \
        --out "arm_cp_$txt.npz"   --meta "meta_cp_$txt.json"
  "$PY" ab_run_arm.py --suffix "_p1_q" --tokens 64 --text "$txt" \
        --out "arm_p1_$txt.npz"   --meta "meta_p1_$txt.json"
  "$PY" ab_compare.py --ref "arm_fp16_$txt.npz" \
        --arm "arm_cp_$txt.npz" --arm "arm_p1_$txt.npz" \
        --meta "meta_fp16_$txt.json" --meta "meta_cp_$txt.json" --meta "meta_p1_$txt.json" \
        --out "p1_compare_$txt.json"
done
"$PY" make_summary.py || true

# 6) 收尾：留现场 → 释放锁 → 关机
echo "--- [6/6] 收尾 $(date '+%F %T')"
free -m | head -2
npu-smi info 2>&1 | sed -n '7p'
ls -l p1_compare_*.json p1_capture_manifest.json "$MODEL_DIR"/layer*_p1_q.om 2>/dev/null | tail -5
echo "P1 产物：$(ls "$MODEL_DIR"/layer*_p1_q.om 2>/dev/null | wc -l)/32 层 + head $([ -f "$MODEL_DIR/head_p1_q.om" ] && echo OK || echo 缺)"

rm -f "$MODEL_DIR/experiment.lock"
pkill -f experiment_guard.sh 2>/dev/null
trap - EXIT
sync
sleep 5
echo "########## 关机 $(date '+%F %T') ##########"
sync
systemctl --no-block poweroff || poweroff || /sbin/poweroff
