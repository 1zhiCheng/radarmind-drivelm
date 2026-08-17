#!/usr/bin/env bash
set -euo pipefail

PY=/home/zhangzongyuan/anaconda3/envs/radargym-rl/bin/python
ROOT=/home/zhangzongyuan/Myproject/drivelm/DriveLM-main
TRAIN=$ROOT/reproduction/qwen_vl_v042_graph/train_graph_ddp.py
BASE=/mnt/data/zzy/drivelm/models/Qwen2.5-VL-7B-Instruct
START=/mnt/data/zzy/drivelm/models/qwen2.5-vl-7b-drivelm-v037b-anchored-dpo-seed42/checkpoint-75
DATA=/mnt/data/zzy/drivelm/reproduction/qwen_vl_v042_graph/manifests/graph_train.jsonl
OUT=/mnt/data/zzy/drivelm/models/qwen2.5-vl-7b-drivelm-v042-graph-sft
RUN=/mnt/data/zzy/drivelm/reproduction/qwen_vl_v042_graph
GPU_5090_0=GPU-9e49353e-f137-3d6d-f7d5-987593a56d30
GPU_5090_2=GPU-07f03596-83e4-920f-9669-fbec78272f67
GPU_5090_3=GPU-8f2fb421-991d-f5eb-c811-448f7e105297

mkdir -p "$OUT" "$RUN/logs"

CUDA_VISIBLE_DEVICES="$GPU_5090_0,$GPU_5090_2,$GPU_5090_3" "$PY" -m accelerate.commands.launch \
  --multi_gpu \
  --num_processes 3 \
  --main_process_port 29642 \
  "$TRAIN" \
  --experiment-name v042-graph-sft-b10shared \
  --model-path "$BASE" \
  --adapter-path "$START" \
  --train-jsonl "$DATA" \
  --output-dir "$OUT" \
  --per-device-batch-size 1 \
  --gradient-accumulation-steps 2 \
  --epochs 1 \
  --max-steps 600 \
  --learning-rate 1e-6 \
  --weight-decay 0 \
  --max-grad-norm 1 \
  --save-steps 100 \
  --max-pixels 100352 \
  --max-length 8192 \
  --seed 42 \
  2>&1 | tee "$RUN/logs/graph_sft.log"

test -s "$OUT/training_report.json"
echo V042_GRAPH_SFT_COMPLETE
