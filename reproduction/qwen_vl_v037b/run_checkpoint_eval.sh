#!/usr/bin/env bash
set -euo pipefail

PY=/home/zhangzongyuan/anaconda3/envs/radargym-rl/bin/python
ROOT=/home/zhangzongyuan/Myproject/drivelm/DriveLM-main
DEV=/mnt/data/zzy/drivelm/reproduction/qwen_vl_stage2_ablation_v1/qwen_v033_c00_dev.jsonl
BASE=/mnt/data/zzy/drivelm/models/Qwen2.5-VL-7B-Instruct
MODEL=/mnt/data/zzy/drivelm/models/qwen2.5-vl-7b-drivelm-v037b-anchored-dpo-seed42
OUT=/mnt/data/zzy/drivelm/reproduction/qwen_vl_v037b/checkpoint_sweep
LOG=/mnt/data/zzy/drivelm/reproduction/qwen_vl_v037b/logs
GPU25=GPU-9e49353e-f137-3d6d-f7d5-987593a56d30
GPU50=GPU-07f03596-83e4-920f-9669-fbec78272f67
GPU75=GPU-8f2fb421-991d-f5eb-c811-448f7e105297
GPU100=GPU-da6b809f-e2bb-bfbe-7f17-8093dd289ed4

mkdir -p "$OUT" "$LOG"
cd "$ROOT"
while tmux has-session -t v037b_train 2>/dev/null; do
  sleep 15
done
test -s "$MODEL/training_report.json"

run_inference() {
  local gpu=$1
  local step=$2
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" reproduction/qwen_vl/infer.py \
    --model-path "$BASE" \
    --adapter-path "$MODEL/checkpoint-$step" \
    --input-jsonl "$DEV" \
    --output-json "$OUT/checkpoint-$step-dev-predictions.json" \
    --device cuda:0 --max-new-tokens 256 --batch-size 16 \
    --max-pixels 100352 --resume \
    > "$LOG/checkpoint-$step-infer.log" 2>&1
}

run_inference "$GPU25" 25 &
PID25=$!
run_inference "$GPU50" 50 &
PID50=$!
run_inference "$GPU75" 75 &
PID75=$!
run_inference "$GPU100" 100 &
PID100=$!
wait "$PID25"
wait "$PID50"
wait "$PID75"
wait "$PID100"

for step in 25 50 75 100; do
  "$PY" reproduction/qwen_vl/evaluate_offline.py \
    --references-jsonl "$DEV" \
    --predictions-json "$OUT/checkpoint-$step-dev-predictions.json" \
    --output-json "$OUT/checkpoint-$step-dev-metrics.json"
done
echo V037B_SWEEP_COMPLETE
