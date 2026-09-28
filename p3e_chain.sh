#!/bin/bash
# P3-E：稠密 Gramian vs 结构化递推（协议 §5 的额外测量）。
# 需要在 CPU 上加载一次 PyTorch 权重（约 6GB），所以必须单独占设备，不能与 NPU 采臂并行。
# 日志：p3e_chain.log；产物 p3e_capture.npz、p3e_gramian.json
#
# 用法：setsid nohup bash p3e_chain.sh < /dev/null > /dev/null 2>&1 &

MODEL_DIR=/home/disk/models/rwkv7-2.9b
LOG=$MODEL_DIR/p3e_chain.log
PY_CPU=/home/disk/miniconda3/envs/rwkv7/bin/python

cd "$MODEL_DIR" || exit 1
exec >> "$LOG" 2>&1

echo
echo "########## P3-E Gramian 一致性/耗时 开始 $(date '+%F %T') ##########"

for p in $(pgrep -f "bash $(basename "$0")"); do
  [ "$p" != "$$" ] && { echo "已有另一份在跑（pid $p），退出"; exit 1; }
done

# 等锁：最多 8 小时；到点仍被占就**退出**，绝不与 NPU 实验并发。
# 9-28 教训：原来"等 80 分钟就硬上"导致 CPU torch(6GB) 与 NPU 引擎(5.5GB) 同时驻留，
# 设备被拖死（负载 17.5、.om 加载报 245000）。
for i in $(seq 1 480); do
  [ -f "$MODEL_DIR/experiment.lock" ] || break
  echo "  等待别的实验链释放 experiment.lock（$i）"
  sleep 60
done
if [ -f "$MODEL_DIR/experiment.lock" ]; then
  echo "锁仍被占用，本链退出（不做并发）"
  exit 1
fi

bash start_service.sh stop
for i in $(seq 1 60); do
  avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  [ "$avail" -gt 6500 ] && break
  sleep 10
done
echo "启动前可用内存 ${avail}MB"

touch "$MODEL_DIR/experiment.lock"
MINE=1
pgrep -f experiment_guard.sh > /dev/null || \
    setsid nohup bash experiment_guard.sh > /dev/null 2>&1 &
trap '[ "$MINE" = "1" ] && rm -f "$MODEL_DIR/experiment.lock"; pkill -f experiment_guard.sh' EXIT

"$PY_CPU" p3e_gramian.py --mode all || echo "P3-E 有失败项"

echo "--- 收尾 $(date '+%F %T')"
free -m | head -2
ls -l p3e_gramian.json 2>/dev/null
rm -f "$MODEL_DIR/experiment.lock"
pkill -f experiment_guard.sh 2>/dev/null
trap - EXIT
sleep 2
bash start_service.sh 8000
echo "########## P3-E 完成（网页服务已恢复）$(date '+%F %T') ##########"
