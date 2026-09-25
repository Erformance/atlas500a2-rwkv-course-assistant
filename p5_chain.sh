#!/bin/bash
# P5 全链：固定工作量微基准（4 臂 × 5 前缀 × 最多 5 次重复）→ 恢复服务 → 真实请求延迟。
# 带单实例检查与内存等待；跑完恢复网页服务并保留开机（是否关机由人决定）。
#
# 日志：p5_chain.log；产物 p5_bench.json、p5_http.json
# 预计 2.5~3 小时（2048 前缀的 prefill 在 fp16 上单次就要 6 分钟）。
#
# 用法：setsid nohup bash p5_chain.sh < /dev/null > /dev/null 2>&1 &

MODEL_DIR=/home/disk/models/rwkv7-2.9b
LOG=$MODEL_DIR/p5_chain.log
PY=/home/disk/miniconda3/envs/npu22/bin/python

cd "$MODEL_DIR" || exit 1
exec >> "$LOG" 2>&1
source /home/disk/cann80base/ascend-toolkit/set_env.sh

echo
echo "########## P5 开始 $(date '+%F %T') ##########"

for p in $(pgrep -f "bash $(basename "$0")"); do
  [ "$p" != "$$" ] && { echo "已有另一份 $(basename "$0") 在跑（pid $p），退出"; exit 1; }
done

bash start_service.sh stop
for i in $(seq 1 24); do
  avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  [ "$avail" -gt 6000 ] && break
  sleep 5
done
echo "启动前可用内存 ${avail}MB"

touch "$MODEL_DIR/experiment.lock"
MINE=1
pgrep -f experiment_guard.sh > /dev/null || \
    setsid nohup bash experiment_guard.sh > /dev/null 2>&1 &
trap '[ "$MINE" = "1" ] && rm -f "$MODEL_DIR/experiment.lock"; pkill -f experiment_guard.sh' EXIT

free -m | head -2
npu-smi info 2>&1 | sed -n '7p'

echo "--- [1/2] 固定工作量微基准 $(date '+%T')"
"$PY" p5_bench.py --arms precise,balanced,fast,full_int8 \
    --prefixes 32,128,512,1024,2048 --decode 128 --repeats 5 \
    --out "$MODEL_DIR/p5_bench.json" || echo "微基准有失败项（可重跑续做）"

echo "--- 微基准完成 $(date '+%T')，恢复服务后测真实请求延迟"
free -m | head -2
rm -f "$MODEL_DIR/experiment.lock"
pkill -f experiment_guard.sh 2>/dev/null
trap - EXIT
sleep 2
bash start_service.sh 8000

echo "--- [2/2] 真实请求延迟（均衡档）$(date '+%T')"
"$PY" p5_http.py --requests 5 --tokens 64 --out "$MODEL_DIR/p5_http.json" || echo "HTTP 延迟失败"

echo "--- 收尾 $(date '+%F %T')"
ls -l p5_bench.json p5_http.json 2>/dev/null
/home/disk/miniconda3/envs/npu22/bin/python -c "
import json
d = json.load(open('p5_bench.json'))
for arm, v in d['arms'].items():
    print('%-10s 加载 %5.1fs｜RSS %7.1fMB｜内存差 %7.1fMB' % (arm, v['load_s_median'], v['rss_mb_median'], v['mem_available_delta_mb_median']))
    for n, p in sorted(v['prefixes'].items(), key=lambda kv: int(kv[0])):
        print('   prefix %-5s prefill %7.3fs｜TTFT %8.1fms｜decode %6.2f tok/s' % (n, p['prefill_s_median'], p['ttft_ms_median'], p['decode_tok_s_median']))
" 2>/dev/null
echo "########## P5 完成 $(date '+%F %T') ##########"
