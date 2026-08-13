#!/usr/bin/env bash
set -euo pipefail

PY=/home/zhangzongyuan/anaconda3/envs/radargym-rl/bin/python
ROOT=/home/zhangzongyuan/Myproject/drivelm/DriveLM-main
GEN=$ROOT/reproduction/qwen_vl_v037a/generate_candidates.py
BASE=/mnt/data/zzy/drivelm/models/Qwen2.5-VL-7B-Instruct
ADAPTER=/mnt/data/zzy/drivelm/models/qwen2.5-vl-7b-drivelm-v037b-anchored-dpo-seed42/checkpoint-75
TRAIN=/mnt/data/zzy/drivelm/reproduction/qwen_vl_v038/important_object_train.jsonl
OUT=/mnt/data/zzy/drivelm/reproduction/qwen_vl_v038/anchor_candidates
LOG=/mnt/data/zzy/drivelm/reproduction/qwen_vl_v038/logs
GPUS=(
  GPU-9e49353e-f137-3d6d-f7d5-987593a56d30
  GPU-07f03596-83e4-920f-9669-fbec78272f67
  GPU-8f2fb421-991d-f5eb-c811-448f7e105297
)

mkdir -p "$OUT" "$LOG"
pids=()
for shard in 0 1 2; do
  (
    CUDA_VISIBLE_DEVICES=${GPUS[$shard]} "$PY" "$GEN" \
      --model-path "$BASE" --adapter-path "$ADAPTER" \
      --train-jsonl "$TRAIN" \
      --output-jsonl "$OUT/v037b_anchor_shard${shard}.jsonl" \
      --num-shards 3 --shard-index "$shard" --batch-size 16 \
      --candidates-per-record 2 --max-new-tokens 256 \
      --temperature 0.8 --top-p 0.9 --repetition-penalty 1.05 \
      --max-pixels 100352 --seed 38042 --resume
  ) > "$LOG/anchor_candidates_shard${shard}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done
echo V038_ANCHOR_CANDIDATES_COMPLETE
