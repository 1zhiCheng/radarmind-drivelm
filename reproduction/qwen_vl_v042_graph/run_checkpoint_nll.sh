#!/usr/bin/env bash
set -euo pipefail

PY=/home/zhangzongyuan/anaconda3/envs/radargym-rl/bin/python
ROOT=/home/zhangzongyuan/Myproject/drivelm/DriveLM-main
BASE=/mnt/data/zzy/drivelm/models/Qwen2.5-VL-7B-Instruct
MODELS=/mnt/data/zzy/drivelm/models/qwen2.5-vl-7b-drivelm-v042-graph-sft
DEV=/mnt/data/zzy/drivelm/reproduction/qwen_vl_v042_graph/manifests/graph_dev.jsonl
OUT=/mnt/data/zzy/drivelm/reproduction/qwen_vl_v042_graph/checkpoint_nll
GPU_5090_0=GPU-9e49353e-f137-3d6d-f7d5-987593a56d30
GPU_5090_2=GPU-07f03596-83e4-920f-9669-fbec78272f67
GPU_5090_3=GPU-8f2fb421-991d-f5eb-c811-448f7e105297
mkdir -p "$OUT"

evaluate_gpu() {
  local gpu=$1
  shift
  for step in "$@"; do
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$ROOT/reproduction/qwen_vl_v042_graph/eval_graph_nll.py" \
      --model-path "$BASE" \
      --adapter-path "$MODELS/checkpoint-$step" \
      --graph-dev-jsonl "$DEV" \
      --output-json "$OUT/checkpoint-${step}.json" \
      --device cuda:0 \
      --batch-size 1 \
      --max-pixels 100352 \
      --max-length 8192 \
      > "$OUT/checkpoint-${step}.log" 2>&1
  done
}

evaluate_gpu "$GPU_5090_0" 100 400 & p0=$!
evaluate_gpu "$GPU_5090_2" 200 500 & p1=$!
evaluate_gpu "$GPU_5090_3" 300 600 & p2=$!
wait "$p0" "$p1" "$p2"

"$PY" "$ROOT/reproduction/qwen_vl_v042_graph/select_nll_checkpoint.py" \
  --reports "$OUT"/checkpoint-*.json \
  --output-json "$OUT/selection.json"

echo V042_GRAPH_NLL_SELECTION_COMPLETE
