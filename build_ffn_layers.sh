#!/bin/bash
# FFN-only int8：只量化每层的 FFN 两个大矩阵（占权重 2/3），
# attention / wkv 循环状态路径保持 fp16 —— 实测状态误差为 0，不会递归放大。
# 产出 layer{i}_fq.om；用法: bash build_ffn_layers.sh <起始层> <结束层>

MODEL_DIR=/home/disk/models/rwkv7-2.9b
CALIB_DIR=$MODEL_DIR/calib
PY_RWKV7=/home/disk/miniconda3/envs/rwkv7/bin/python
PY_QUANT=/home/disk/miniconda3/envs/quant/bin/python
CANN_ENV=/home/disk/cann80base/ascend-toolkit/set_env.sh
TBE_PY=/home/disk/miniconda3/envs/npu22/bin
INCLUDE="/ffn/key/MatMul,/ffn/value/MatMul"
METHOD=${CALIB_METHOD:-1}
MEM_MIN_MB=${MEM_MIN_MB:-3500}

wait_mem() {
  local avail
  avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  while [ "$avail" -lt "$MEM_MIN_MB" ]; do
    echo "[$(date +%H:%M:%S)] 可用内存 ${avail}MB < ${MEM_MIN_MB}MB，等待 30s"
    sleep 30
    avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  done
}

for i in $(seq "$1" "$2"); do
  n=$(printf "%02d" "$i")
  src="$MODEL_DIR/layer${i}.onnx"
  sim="$MODEL_DIR/layer${i}_sim.onnx"
  cq="$MODEL_DIR/layer${i}_fq.onnx"
  om="$MODEL_DIR/layer${i}_fq.om"

  if [ -f "$om" ]; then
    echo "[$(date +%H:%M:%S)] layer$i 跳过（已有 _fq 模型）"
    continue
  fi

  echo "[$(date +%H:%M:%S)] layer$i 开始（$INCLUDE）"
  if [ ! -f "$sim" ]; then
    nice -n 19 "$PY_RWKV7" -c "
import onnx
from onnxsim import simplify
m = onnx.load('$src')
sm, ok = simplify(m)
onnx.save(sm, '$sim')
print('simplify ok', ok, len(sm.graph.node))
" > "$MODEL_DIR/simplify_fq_layer${i}.log" 2>&1
    [ -f "$sim" ] || { echo "[$(date +%H:%M:%S)] layer$i 简化失败"; continue; }
  fi

  nice -n 19 "$PY_QUANT" "$MODEL_DIR/quantize_layer_sel.py" "$sim" "$cq" \
      --calib "$CALIB_DIR/layer${n}.npz" --include "$INCLUDE" --method "$METHOD" \
      > "$MODEL_DIR/quant_fq_layer${i}.log" 2>&1
  [ -s "$cq" ] || { echo "[$(date +%H:%M:%S)] layer$i 量化失败"; continue; }
  echo "[$(date +%H:%M:%S)] layer$i 量化完成 $(du -h "$cq" | cut -f1)"

  wait_mem
  (
    . "$CANN_ENV"
    export PATH="$TBE_PY:$PATH"
    export TE_PARALLEL_COMPILER=1
    nice -n 19 atc --model="$cq" --framework=5 --output="${om%.om}" \
        --soc_version=Ascend310B1 --input_format=ND --log=error
  ) > "$MODEL_DIR/atc_fq_layer${i}.log" 2>&1

  if [ -f "$om" ]; then
    rm -f "$cq" "$sim"
    echo "[$(date +%H:%M:%S)] layer$i OK $(du -h "$om" | cut -f1)"
  else
    echo "[$(date +%H:%M:%S)] layer$i ATC 失败（见 atc_fq_layer${i}.log）"
  fi
done
echo "worker $1-$2 done"
