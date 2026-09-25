#!/bin/bash
# P1 重建：用扩展校准数据（calib_p1/，96 组 × 状态年龄 0/64/256）重做 32 层 + 输出头。
# 产出 layer{i}_p1_q.om / head_p1_q.om；已有产物自动跳过。
# 与 _cp_q（旧校准）并存，互不覆盖，方便直接对照。
#
# 用法: bash build_p1_layers.sh [层号列表]
#   bash build_p1_layers.sh            # 全部 32 层 + head
#   bash build_p1_layers.sh 0 9 21     # 只做这几层（快速试点）
#
# 注意：实验期间由调用方持有 experiment.lock（p1_chain.sh 负责），本脚本不碰锁。

MODEL_DIR=/home/disk/models/rwkv7-2.9b
PY_RWKV7=/home/disk/miniconda3/envs/rwkv7/bin/python
PY_QUANT=/home/disk/miniconda3/envs/quant/bin/python
CANN_ENV=/home/disk/cann80base/ascend-toolkit/set_env.sh
TBE_PY=/home/disk/miniconda3/envs/npu22/bin
# 可用环境变量覆盖，方便做对照臂（例如 P2-B 的等样本数实验）：
#   CALIB=.../calib_b0 SUFFIX=_b0_q LOG_PREFIX=b0 BUILD_HEAD=0 bash build_p1_layers.sh 24 6 11 …
CALIB=${CALIB:-$MODEL_DIR/calib_p1}
SUFFIX=${SUFFIX:-_p1_q}
LOG_PREFIX=${LOG_PREFIX:-p1}
BUILD_HEAD=${BUILD_HEAD:-1}

if [ "$#" -gt 0 ]; then
  LAYERS="$@"
else
  LAYERS=$(seq 0 31)
fi

simplify() {   # $1=层号
  local i=$1 src="$MODEL_DIR/layer${1}.onnx" sim="$MODEL_DIR/layer${1}_sim.onnx"
  [ -f "$sim" ] && return 0
  nice -n 19 "$PY_RWKV7" -c "
import onnx
from onnxsim import simplify
m = onnx.load('$src'); sm, ok = simplify(m); onnx.save(sm, '$sim')
print('simplify ok', ok)
" > "$MODEL_DIR/simplify_${LOG_PREFIX}_layer${i}.log" 2>&1
  [ -f "$sim" ]
}

quantize() {   # $1=名字  $2=sim  $3=out  $4=npz
  nice -n 19 "$PY_QUANT" "$MODEL_DIR/quantize_layer_calib.py" "$2" "$3" \
      --calib "$CALIB/$4" --method 1 > "$MODEL_DIR/quant_${LOG_PREFIX}_$1.log" 2>&1
  [ -s "$3" ]
}

compile_om() { # $1=onnx  $2=out前缀
  (
    . "$CANN_ENV"
    export PATH="$TBE_PY:$PATH"
    export TE_PARALLEL_COMPILER=1
    nice -n 19 atc --model="$1" --framework=5 --output="$2" \
        --soc_version=Ascend310B1 --input_format=ND --log=error
  ) > "$MODEL_DIR/atc_${LOG_PREFIX}_$(basename "$2").log" 2>&1
  [ -f "$2.om" ]
}

for i in $LAYERS; do
  om="$MODEL_DIR/layer${i}${SUFFIX}.om"
  q="$MODEL_DIR/layer${i}${SUFFIX}.onnx"
  if [ -f "$om" ]; then echo "[$(date +%H:%M:%S)] layer$i 跳过（已有）"; continue; fi
  echo "[$(date +%H:%M:%S)] === layer$i"
  simplify "$i" || { echo "  简化失败"; continue; }
  if [ ! -s "$q" ]; then
    echo "  量化（$(basename "$CALIB")/layer$(printf %02d $i).npz）…"
    quantize "layer${i}" "$MODEL_DIR/layer${i}_sim.onnx" "$q" "layer$(printf %02d $i).npz" \
        || { echo "  量化失败"; continue; }
  fi
  echo "  ATC 编译…"
  compile_om "$q" "$MODEL_DIR/layer${i}${SUFFIX}" || { echo "  ATC 失败"; continue; }
  rm -f "$q"
  echo "[$(date +%H:%M:%S)] layer$i OK"
done

# 输出头：head 的输入是第 31 层的 x 输出，用同一份校准数据重跑一遍 fp16 layer31 得到
if [ "$BUILD_HEAD" = "1" ] && [ ! -f "$MODEL_DIR/head${SUFFIX}.om" ]; then
  echo "[$(date +%H:%M:%S)] === head"
  if [ ! -f "$CALIB/head.npz" ]; then
    . "$CANN_ENV"
    "$TBE_PY/python" "$MODEL_DIR/make_head_calib.py" --calib-dir "$CALIB" || exit 1
  fi
  [ -s "$MODEL_DIR/head${SUFFIX}.onnx" ] || quantize head "$MODEL_DIR/head_sim.onnx" \
      "$MODEL_DIR/head${SUFFIX}.onnx" head.npz
  compile_om "$MODEL_DIR/head${SUFFIX}.onnx" "$MODEL_DIR/head${SUFFIX}" \
      && rm -f "$MODEL_DIR/head${SUFFIX}.onnx"
fi

echo "[$(date +%H:%M:%S)] ${LOG_PREFIX} 完成：$(ls "$MODEL_DIR"/layer*${SUFFIX}.om 2>/dev/null | wc -l)/32 层，head $([ -f "$MODEL_DIR/head${SUFFIX}.om" ] && echo OK || echo 缺)"
