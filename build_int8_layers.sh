#!/bin/bash
# 逐层构建 int8 模型（导出 → 简化 → 量化 → ATC 编译），已有 layer{i}_q.om 的层自动跳过。
# 用法: bash build_int8_layers.sh <起始层> <结束层>
#       bash build_int8_layers.sh head            # 只做输出头 head_q.om

MODEL_DIR=/home/disk/models/rwkv7-2.9b
PY_RWKV7=/home/disk/miniconda3/envs/rwkv7/bin/python     # 导出 + onnx-simplify
PY_QUANT=/home/disk/miniconda3/envs/quant/bin/python      # modelslim 量化
CANN_ENV=/home/disk/cann80base/ascend-toolkit/set_env.sh
TBE_PY=/home/disk/miniconda3/envs/npu22/bin
MEM_MIN_MB=${MEM_MIN_MB:-3500}   # ATC 编译前的可用内存闸门

# 内存闸门：这台机器 11GB 内存且没有 swap，ATC 同时跑多个会把整机压死
wait_mem() {
  local avail
  avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  while [ "$avail" -lt "$MEM_MIN_MB" ]; do
    echo "[$(date +%H:%M:%S)] 可用内存 ${avail}MB < ${MEM_MIN_MB}MB，等待 30s"
    sleep 30
    avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  done
}

# 公共的一步：简化 → 量化 → ATC（layer 与 head 共用）
build_one() {
  local name="$1"          # 例如 layer7 / head
  local src="$MODEL_DIR/${name}.onnx"
  local sim="$MODEL_DIR/${name}_sim.onnx"
  local q="$MODEL_DIR/${name}_q.onnx"
  local om="$MODEL_DIR/${name}_q.om"

  echo "[$(date +%H:%M:%S)] $name 开始（源 $(du -h "$src" | cut -f1)）"

  if [ ! -f "$sim" ]; then
    nice -n 19 "$PY_RWKV7" -c "
import onnx
from onnxsim import simplify
m = onnx.load('$src')
sm, ok = simplify(m)
onnx.save(sm, '$sim')
print('simplify ok', ok, len(sm.graph.node))
" > "$MODEL_DIR/simplify_${name}.log" 2>&1
    [ -f "$sim" ] || { echo "[$(date +%H:%M:%S)] $name 简化失败"; return 1; }
  fi
  echo "[$(date +%H:%M:%S)] $name 简化完成"

  if [ ! -f "$q" ]; then
    nice -n 19 "$PY_QUANT" "$MODEL_DIR/quantize_layer.py" "$sim" "$q" \
        > "$MODEL_DIR/quant_${name}.log" 2>&1
    [ -f "$q" ] || { echo "[$(date +%H:%M:%S)] $name 量化失败（见 quant_${name}.log）"; return 1; }
    echo "[$(date +%H:%M:%S)] $name 量化完成 $(du -h "$q" | cut -f1)"
  else
    echo "[$(date +%H:%M:%S)] $name 复用已有量化结果 $(du -h "$q" | cut -f1)"
  fi

  wait_mem
  echo "[$(date +%H:%M:%S)] $name 开始 ATC（可用内存 $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)MB）"
  (
    . "$CANN_ENV"
    export PATH="$TBE_PY:$PATH"
    export TE_PARALLEL_COMPILER=1
    nice -n 19 atc --model="$q" --framework=5 --output="${om%.om}" \
        --soc_version=Ascend310B1 --input_format=ND --log=error
  ) > "$MODEL_DIR/atc_q_${name}.log" 2>&1

  if [ -f "$om" ]; then
    echo "[$(date +%H:%M:%S)] $name OK $(du -h "$om" | cut -f1)"
    return 0
  fi
  echo "[$(date +%H:%M:%S)] $name ATC 失败（见 atc_q_${name}.log）"
  return 1
}

if [ "$1" = "head" ]; then
  build_one head
  echo "head done"
  exit 0
fi

for i in $(seq "$1" "$2"); do
  if [ -f "$MODEL_DIR/layer${i}_q.om" ]; then
    echo "[$(date +%H:%M:%S)] layer $i 跳过（已有 int8 模型）"
    continue
  fi
  if [ ! -f "$MODEL_DIR/layer${i}.onnx" ]; then
    nice -n 19 "$PY_RWKV7" "$MODEL_DIR/rwkv7_export_chunk.py" --start "$i" --layers 1 \
        --dtype float32 --out "$MODEL_DIR/layer${i}.onnx" > "$MODEL_DIR/export_layer${i}.log" 2>&1
    [ -f "$MODEL_DIR/layer${i}.onnx" ] || { echo "layer $i 导出失败"; continue; }
  fi
  build_one "layer${i}"
done

# 每层产出的中间文件很大（fp32 ONNX 300MB+），成功后删掉省空间
for i in $(seq "$1" "$2"); do
  if [ -f "$MODEL_DIR/layer${i}_q.om" ]; then
    rm -f "$MODEL_DIR/layer${i}_q.onnx" "$MODEL_DIR/layer${i}_sim.onnx"
  fi
done
echo "worker $1-$2 done"
