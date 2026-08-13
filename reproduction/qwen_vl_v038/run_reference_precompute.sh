#!/usr/bin/env bash
set -euo pipefail

PY=/home/zhangzongyuan/anaconda3/envs/radargym-rl/bin/python
ROOT=/home/zhangzongyuan/Myproject/drivelm/DriveLM-main
SCRIPT=$ROOT/reproduction/qwen_vl_v038/precompute_reference.py
MERGE=$ROOT/reproduction/qwen_vl_v038/merge_reference_shards.py
BASE=/mnt/data/zzy/drivelm/models/Qwen2.5-VL-7B-Instruct
ADAPTER=/mnt/data/zzy/drivelm/models/qwen2.5-vl-7b-drivelm-v037b-anchored-dpo-seed42/checkpoint-75
PREF=/mnt/data/zzy/drivelm/reproduction/qwen_vl_v038/grounding_balanced_preferences.jsonl
OUT=/mnt/data/zzy/drivelm/reproduction/qwen_vl_v038/reference
LOG=/mnt/data/zzy/drivelm/reproduction/qwen_vl_v038/logs
GPUS=(GPU-9e49353e-f137-3d6d-f7d5-987593a56d30 GPU-07f03596-83e4-920f-9669-fbec78272f67 GPU-8f2fb421-991d-f5eb-c811-448f7e105297)

mkdir -p "$OUT" "$LOG"
pids=()
for shard in 0 1 2; do
  (
    CUDA_VISIBLE_DEVICES=${GPUS[$shard]} "$PY" "$SCRIPT" \
      --model-path "$BASE" --adapter-path "$ADAPTER" \
      --preference-jsonl "$PREF" \
      --output-jsonl "$OUT/reference_shard${shard}.jsonl" \
      --num-shards 3 --shard-index "$shard" --resume \
      --reference-policy-label 'v0.37B checkpoint-75' \
      --max-pixels 100352 --max-length 4096 --normalization sum
  ) > "$LOG/reference_shard${shard}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

"$PY" "$MERGE" \
  --preference-jsonl "$PREF" \
  --reference-jsonl "$OUT/reference_shard0.jsonl" \
  --reference-jsonl "$OUT/reference_shard1.jsonl" \
  --reference-jsonl "$OUT/reference_shard2.jsonl" \
  --output-jsonl "$OUT/preferences_with_v037b_reference.jsonl"

echo V038_REFERENCE_COMPLETE
