#!/bin/bash
# 文本质量对照：fp16 / top8 / top16 三档，各问"自然问答"和"代码题"一道。
# 目的：判断 top8 能否作为默认平衡档（P3 帕累托里 1.19× 且 KL=0.019）。
#
# 用法（在设备上，脱离终端跑）：
#   cd /home/disk/models/rwkv7-2.9b
#   setsid nohup bash quality_cmp.sh < /dev/null > /dev/null 2>&1 &
#   输出落在 quality_cmp.log
#
# 前提：experiment.lock 存在（挡住 5 分钟看门狗抢 NPU）、机器上只有这一个模型实例。

cd /home/disk/models/rwkv7-2.9b || exit 1
LOG=${LOG:-quality_cmp.log}
exec > "$LOG" 2>&1

source /home/disk/cann80base/ascend-toolkit/set_env.sh
PY=/home/disk/miniconda3/envs/npu22/bin/python
export PYTHONUNBUFFERED=1

TOKENS=${TOKENS:-160}
STAGES=${STAGES:-"fp16 top8 top16"}      # 可只跑其中几档，例如 STAGES=fp16
QS=${QS:-"什么是熵？|写一个判断素数的Python函数"}   # 用 | 分隔多个问题

echo "##### 开始 $(date '+%F %T') ｜ 每题 ${TOKENS} token #####"

run() {
  name="$1"; shift
  echo
  echo "==================== [$name] $(date '+%T') ===================="
  {
    echo "${QS%%|*}"
    echo "/reset"
    [ "${QS#*|}" != "$QS" ] && echo "${QS#*|}"
    echo "/quit"
  } | $PY rwkv7_chat.py --serve --tokens "$TOKENS" "$@" \
        2>&1 | grep --line-buffered -v -E '^\[(INFO|EVENT|WARNING|ERROR)\]'
  echo "[$name] 结束 $(date '+%T')"
}

case " $STAGES " in *" fp16 "*)  run fp16 ;; esac
case " $STAGES " in *" top8 "*)  run top8  --plan-file plan_top8.json ;; esac
case " $STAGES " in *" top16 "*) run top16 --plan-file plan_top16.json ;; esac

echo
echo "##### ALLDONE $(date '+%F %T') #####"
