#!/bin/bash
# P2-A 全栈重建：用"执行前采样"的校准数据（calib_pre/）重做 32 层 + 输出头。
# 产出 layer{i}_cp_q.om / head_cp_q.om；已有产物自动跳过。
# 用法: bash build_cp_layers.sh

MODEL_DIR=/home/disk/models/rwkv7-2.9b
PY_RWKV7=/home/disk/miniconda3/envs/rwkv7/bin/python
PY_QUANT=/home/disk/miniconda3/envs/quant/bin/python
CANN_ENV=/home/disk/cann80base/ascend-toolkit/set_env.sh
TBE_PY=/home/disk/miniconda3/envs/npu22/bin
CALIB=$MODEL_DIR/calib_pre

# 实验期间独占 NPU：给看门狗上锁（它每 5 分钟跑一次，会与实验抢设备）
touch "$MODEL_DIR/experiment.lock"
trap 'true' EXIT

simplify() {   # $1=层号
  local i=$1 src="$MODEL_DIR/layer${1}.onnx" sim="$MODEL_DIR/layer${1}_sim.onnx"
  [ -f "$sim" ] && return 0
  nice -n 19 "$PY_RWKV7" -c "
import onnx
from onnxsim import simplify
m = onnx.load('$src'); sm, ok = simplify(m); onnx.save(sm, '$sim')
print('simplify ok', ok)
" > "$MODEL_DIR/simplify_cp_layer${i}.log" 2>&1
  [ -f "$sim" ]
}

quantize() {   # $1=名字(如 layer7 / head)  $2=sim  $3=out
  nice -n 19 "$PY_QUANT" "$MODEL_DIR/quantize_layer_calib.py" "$2" "$3" \
      --calib "$CALIB/$4" --method 1 > "$MODEL_DIR/quant_cp_$1.log" 2>&1
  [ -s "$3" ]
}

compile_om() { # $1=onnx  $2=out前缀
  (
    . "$CANN_ENV"
    export PATH="$TBE_PY:$PATH"
    export TE_PARALLEL_COMPILER=1
    nice -n 19 atc --model="$1" --framework=5 --output="$2" \
        --soc_version=Ascend310B1 --input_format=ND --log=error
  ) > "$MODEL_DIR/atc_cp_$(basename "$2").log" 2>&1
  [ -f "$2.om" ]
}

for i in $(seq 0 31); do
  om="$MODEL_DIR/layer${i}_cp_q.om"
  q="$MODEL_DIR/layer${i}_cp_q.onnx"
  if [ -f "$om" ]; then echo "[$(date +%H:%M:%S)] layer$i 跳过（已有）"; continue; fi
  echo "[$(date +%H:%M:%S)] === layer$i"
  simplify "$i" || { echo "  简化失败"; continue; }
  [ -s "$q" ] || { echo "  量化…"; quantize "layer${i}" "$MODEL_DIR/layer${i}_sim.onnx" "$q" "layer$(printf %02d $i).npz" \
      || { echo "  量化失败"; continue; }; }
  echo "  ATC 编译…"
  compile_om "$q" "$MODEL_DIR/layer${i}_cp_q" || { echo "  ATC 失败"; continue; }
  rm -f "$q"
  echo "[$(date +%H:%M:%S)] layer$i OK"
done

# 输出头
if [ ! -f "$MODEL_DIR/head_cp_q.om" ]; then
  echo "[$(date +%H:%M:%S)] === head"
  if [ ! -f "$CALIB/head.npz" ]; then
    source "$CANN_ENV"
    "$TBE_PY/python" "$MODEL_DIR/make_head_calib.py" --calib-dir "$CALIB" || exit 1
  fi
  [ -s "$MODEL_DIR/head_cp_q.onnx" ] || quantize head "$MODEL_DIR/head_sim.onnx" "$MODEL_DIR/head_cp_q.onnx" head.npz
  compile_om "$MODEL_DIR/head_cp_q.onnx" "$MODEL_DIR/head_cp_q" && rm -f "$MODEL_DIR/head_cp_q.onnx"
fi

rm -f "$MODEL_DIR/experiment.lock"
echo "[$(date +%H:%M:%S)] 全部完成：$(ls "$MODEL_DIR"/layer*_cp_q.om 2>/dev/null | wc -l)/32 层"
