#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-/home/zhangzongyuan/anaconda3/envs/radargym-rl/bin/python}
ROOT=${ROOT:-/home/zhangzongyuan/Myproject/drivelm/DriveLM-main}
BASE=${BASE:-/mnt/data/zzy/drivelm/models/Qwen2.5-VL-7B-Instruct}
START=${START:-/mnt/data/zzy/drivelm/models/qwen2.5-vl-7b-drivelm-v037b-anchored-dpo-seed42/checkpoint-75}
GRAPH=${GRAPH:-/mnt/data/zzy/drivelm/reproduction/qwen_vl_v042_graph/manifests/graph_train.jsonl}
QA=${QA:-/mnt/data/zzy/drivelm/reproduction/qwen_vl_stage2_ablation_v1/qwen_v033_c00_train.jsonl}
RUN=${RUN:-/mnt/data/zzy/drivelm/reproduction/qwen_vl_v042_graph_anchor}
OUT=${OUT:-/mnt/data/zzy/drivelm/models/qwen2.5-vl-7b-drivelm-v042b-graph-anchor}
GPU_5090_0=${GPU_5090_0:-GPU-9e49353e-f137-3d6d-f7d5-987593a56d30}
GPU_5090_2=${GPU_5090_2:-GPU-07f03596-83e4-920f-9669-fbec78272f67}
GPU_5090_3=${GPU_5090_3:-GPU-8f2fb421-991d-f5eb-c811-448f7e105297}

mkdir -p "$RUN/manifests" "$RUN/logs" "$OUT"
"$PY" "$ROOT/reproduction/qwen_vl_v042_graph/build_graph_anchor_mixture.py" \
  --graph-jsonl "$GRAPH" \
  --independent-jsonl "$QA" \
  --output-jsonl "$RUN/manifests/graph_anchor50_train.jsonl" \
  --report-json "$RUN/manifests/manifest_report.json" \
  --anchors-per-task 901 --seed 42

CUDA_VISIBLE_DEVICES="$GPU_5090_0,$GPU_5090_2,$GPU_5090_3" "$PY" -m accelerate.commands.launch \
  --multi_gpu --num_processes 3 --main_process_port 29643 \
  "$ROOT/reproduction/qwen_vl_v042_graph/train_graph_ddp.py" \
  --experiment-name v042b-graph-anchor50-b10shared \
  --model-path "$BASE" --adapter-path "$START" \
  --train-jsonl "$RUN/manifests/graph_anchor50_train.jsonl" \
  --output-dir "$OUT" \
  --per-device-batch-size 1 --gradient-accumulation-steps 2 \
  --epochs 1 --max-steps 600 --learning-rate 1e-6 \
  --weight-decay 0 --max-grad-norm 1 --save-steps 100 \
  --max-pixels 100352 --max-length 8192 --seed 42 \
  2>&1 | tee "$RUN/logs/train.log"

test -s "$OUT/training_report.json"
echo V042B_GRAPH_ANCHOR_COMPLETE
