#!/bin/bash
# 用法: bash rwkv7_build_layers.sh <start_layer> <stop_layer>
# 逐层导出 ONNX 并用 ATC 编译成 .om；已有 .om 的层自动跳过。

MODEL_DIR=/home/disk/models/rwkv7-2.9b
CANN_ENV=/home/disk/cann80base/ascend-toolkit/set_env.sh
PY_EXPORT=/home/disk/miniconda3/envs/rwkv7/bin/python
PY_TBE=/home/disk/miniconda3/envs/npu22/bin

for i in $(seq "$1" "$2"); do
  echo "=== layer $i"
  if [ -f "$MODEL_DIR/layer${i}.om" ]; then
    echo "layer $i SKIP (已有 .om)"
    continue
  fi
  nice -n 19 "$PY_EXPORT" "$MODEL_DIR/rwkv7_export_chunk.py" --start "$i" --layers 1 \
      --dtype float32 --out "$MODEL_DIR/layer${i}.onnx" > "$MODEL_DIR/export_layer${i}.log" 2>&1
  if [ ! -f "$MODEL_DIR/layer${i}.onnx" ]; then
    echo "layer $i EXPORT_FAILED"
    continue
  fi
  . "$CANN_ENV"
  export PATH="$PY_TBE:$PATH"
  TE_PARALLEL_COMPILER=1 nice -n 19 atc --model="$MODEL_DIR/layer${i}.onnx" --framework=5 \
      --output="$MODEL_DIR/layer${i}" --soc_version=Ascend310B1 --input_format=ND \
      --log=error > "$MODEL_DIR/atc_layer${i}.log" 2>&1
  if [ -f "$MODEL_DIR/layer${i}.om" ]; then
    echo "layer $i OK"
  else
    echo "layer $i FAILED"
  fi
done
echo "worker $1-$2 done"
