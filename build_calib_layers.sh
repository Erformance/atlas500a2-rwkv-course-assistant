#!/bin/bash
# 用真实校准数据重做 int8 模型（产出 layer{i}_cq.om / head_cq.om）。
# 与 label-free 版共用同一套流水线，只多了 --calib。
# 用法: bash build_calib_layers.sh <起始层> <结束层>
#       bash build_calib_layers.sh head
# 环境变量: CALIB_METHOD（0 min-max / 1 percentile / 2 entropy，默认 1）

MODEL_DIR=/home/disk/models/rwkv7-2.9b
CALIB_DIR=$MODEL_DIR/calib
PY_RWKV7=/home/disk/miniconda3/envs/rwkv7/bin/python
PY_QUANT=/home/disk/miniconda3/envs/quant/bin/python
CANN_ENV=/home/disk/cann80base/ascend-toolkit/set_env.sh
TBE_PY=/home/disk/miniconda3/envs/npu22/bin
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

build_one() {
  local name="$1"          # layer7 / head
  local calib="$2"         # 校准 npz
  local src="$MODEL_DIR/${name}.onnx"
  local sim="$MODEL_DIR/${name}_sim.onnx"
  local cq="$MODEL_DIR/${name}_cq.onnx"
  local om="$MODEL_DIR/${name}_cq.om"

  echo "[$(date +%H:%M:%S)] $name 开始（源 $(du -h "$src" | cut -f1)，校准 $(basename "$calib")，method=$METHOD）"

  if [ ! -f "$sim" ]; then
    nice -n 19 "$PY_RWKV7" -c "
import onnx
from onnxsim import simplify
m = onnx.load('$src')
sm, ok = simplify(m)
onnx.save(sm, '$sim')
print('simplify ok', ok, len(sm.graph.node))
" > "$MODEL_DIR/simplify_cq_${name}.log" 2>&1
    [ -f "$sim" ] || { echo "[$(date +%H:%M:%S)] $name 简化失败"; return 1; }
  fi
  echo "[$(date +%H:%M:%S)] $name 简化完成"

  nice -n 19 "$PY_QUANT" "$MODEL_DIR/quantize_layer_calib.py" "$sim" "$cq" \
      --calib "$calib" --method "$METHOD" > "$MODEL_DIR/quant_cq_${name}.log" 2>&1
  [ -s "$cq" ] || { echo "[$(date +%H:%M:%S)] $name 校准量化失败（见 quant_cq_${name}.log）"; return 1; }
  echo "[$(date +%H:%M:%S)] $name 量化完成 $(du -h "$cq" | cut -f1)"

  wait_mem
  echo "[$(date +%H:%M:%S)] $name 开始 ATC（可用内存 $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)MB）"
  (
    . "$CANN_ENV"
    export PATH="$TBE_PY:$PATH"
    export TE_PARALLEL_COMPILER=1
    nice -n 19 atc --model="$cq" --framework=5 --output="${om%.om}" \
        --soc_version=Ascend310B1 --input_format=ND --log=error
  ) > "$MODEL_DIR/atc_cq_${name}.log" 2>&1

  if [ -f "$om" ]; then
    rm -f "$cq" "$sim"
    echo "[$(date +%H:%M:%S)] $name OK $(du -h "$om" | cut -f1)"
    return 0
  fi
  echo "[$(date +%H:%M:%S)] $name ATC 失败（见 atc_cq_${name}.log）"
  return 1
}

if [ "$1" = "head" ]; then
  build_one head "$CALIB_DIR/head.npz"
  echo "head done"
  exit 0
fi

for i in $(seq "$1" "$2"); do
  n=$(printf "%02d" "$i")
  if [ -f "$MODEL_DIR/layer${i}_cq.om" ]; then
    echo "[$(date +%H:%M:%S)] layer$i 跳过（已有 _cq 模型）"
    continue
  fi
  build_one "layer$i" "$CALIB_DIR/layer${n}.npz"
done
echo "worker $1-$2 done"
