#!/bin/bash
# 补跑 P4 ① 的 NPU 侧（上次因 /loglikelihood_detail 路径判断顺序写错而失败），然后与 CPU 结果比对。
MODEL_DIR=/home/disk/models/rwkv7-2.9b
LOG=$MODEL_DIR/p4_token_npu_rerun.log
PY=/home/disk/miniconda3/envs/npu22/bin/python

cd "$MODEL_DIR" || exit 1
exec >> "$LOG" 2>&1
source /home/disk/cann80base/ascend-toolkit/set_env.sh

echo "########## NPU 侧补跑 $(date '+%F %T') ##########"
bash start_service.sh stop
pgrep -f p4_score_server.py > /dev/null && { pkill -f p4_score_server.py; sleep 3; }
touch "$MODEL_DIR/experiment.lock"
MINE=1
trap '[ "$MINE" = "1" ] && rm -f "$MODEL_DIR/experiment.lock"' EXIT

setsid nohup "$PY" p4_score_server.py --port 8100 > p4_score_token_check.log 2>&1 < /dev/null &
for i in $(seq 1 30); do
  sleep 5
  grep -q "服务已就绪" p4_score_token_check.log 2>/dev/null && break
done
tail -2 p4_score_token_check.log

"$PY" p4_token_check.py --mode npu --ctx $'Question: How do you close a door?\nAnswer:' \
    --cont " Turn the handle and push." --max-tokens 4 --out p4_token_npu.json || echo "NPU 侧失败"
"$PY" p4_token_check.py --mode npu --ctx "The capital of France is" \
    --cont " Paris" --max-tokens 2 --out p4_token_npu_b.json || echo "NPU 短句失败"
pkill -f p4_score_server.py; sleep 3

echo "--- 比对 ---"
"$PY" p4_token_check.py --mode compare --compare p4_token_npu.json p4_token_cpu.json \
    --out p4_token_vs.json || echo "比对失败"
"$PY" p4_token_check.py --mode compare --compare p4_token_npu_b.json p4_token_cpu_b.json \
    --out p4_token_vs_b.json || echo "短句比对失败"

rm -f "$MODEL_DIR/experiment.lock"
trap - EXIT
bash start_service.sh 8000
echo "########## 补跑完成（网页服务已恢复）$(date '+%F %T') ##########"
