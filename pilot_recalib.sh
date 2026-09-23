#!/bin/bash
# P2-A 小试：同一层用"执行前采样"的校准数据重新量化，与旧数据交叉对比。
# 产出 layer{i}_cp_q.om（cp = calibration, pre-execution sampling）
# 用法: bash pilot_recalib.sh 0 31

MODEL_DIR=/home/disk/models/rwkv7-2.9b
PY_RWKV7=/home/disk/miniconda3/envs/rwkv7/bin/python
PY_QUANT=/home/disk/miniconda3/envs/quant/bin/python
CANN_ENV=/home/disk/cann80base/ascend-toolkit/set_env.sh
TBE_PY=/home/disk/miniconda3/envs/npu22/bin

for i in "$@"; do
  n=$(printf "%02d" "$i")
  sim="$MODEL_DIR/layer${i}_sim.onnx"
  q="$MODEL_DIR/layer${i}_cp_q.onnx"
  om="$MODEL_DIR/layer${i}_cp_q.om"

  echo "=== layer$i"
  if [ ! -f "$sim" ]; then
    echo "  简化 ONNX …"
    nice -n 19 "$PY_RWKV7" -c "
import onnx
from onnxsim import simplify
m = onnx.load('$MODEL_DIR/layer${i}.onnx')
sm, ok = simplify(m)
onnx.save(sm, '$sim')
print('simplify ok', ok)
" > "$MODEL_DIR/simplify_cp_layer${i}.log" 2>&1
    [ -f "$sim" ] || { echo "  简化失败"; continue; }
  fi

  if [ ! -s "$q" ]; then
    echo "  用新时序数据量化（calib_pre/layer${n}.npz）…"
    nice -n 19 "$PY_QUANT" "$MODEL_DIR/quantize_layer_calib.py" "$sim" "$q" \
        --calib "$MODEL_DIR/calib_pre/layer${n}.npz" --method 1 \
        > "$MODEL_DIR/quant_cp_layer${i}.log" 2>&1
    [ -s "$q" ] || { echo "  量化失败（见 quant_cp_layer${i}.log）"; continue; }
    echo "  量化完成 $(du -h "$q" | cut -f1)"
  fi

  if [ ! -f "$om" ]; then
    echo "  ATC 编译 …"
    (
      . "$CANN_ENV"
      export PATH="$TBE_PY:$PATH"
      export TE_PARALLEL_COMPILER=1
      nice -n 19 atc --model="$q" --framework=5 --output="${om%.om}" \
          --soc_version=Ascend310B1 --input_format=ND --log=error
    ) > "$MODEL_DIR/atc_cp_layer${i}.log" 2>&1
    [ -f "$om" ] || { echo "  ATC 失败（见 atc_cp_layer${i}.log）"; continue; }
  fi
  echo "  OK $(du -h "$om" | cut -f1)"
done

# ---- 对比：fp16 vs label-free vs 旧校准 vs 新校准（同一批真实输入）----
echo
echo "================= 误差对比（用 calib_pre 的真实输入）================"
source "$CANN_ENV"
for i in "$@"; do
  n=$(printf "%02d" "$i")
  echo "### layer$i"
  for v in "label-free:layer${i}_q.om" "旧校准:layer${i}_cq.om" "新校准(pre):layer${i}_cp_q.om"; do
    name="${v%%:*}"; f="${v##*:}"
    if [ -f "$MODEL_DIR/$f" ]; then
      printf "  %-12s " "$name"
      "$TBE_PY/python" "$MODEL_DIR/cmp_layer_real.py" "$MODEL_DIR/layer${i}.om" \
          "$MODEL_DIR/$f" "$MODEL_DIR/calib_pre/layer${n}.npz" 2>&1 | grep 最大
    else
      printf "  %-12s （无 %s）\n" "$name" "$f"
    fi
  done
done
echo "==================================================================="
