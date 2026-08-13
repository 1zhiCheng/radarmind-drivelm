#!/usr/bin/env bash
set -euo pipefail

PY=/home/zhangzongyuan/anaconda3/envs/radargym-rl/bin/python
ROOT=/home/zhangzongyuan/Myproject/drivelm/DriveLM-main
WORK=/mnt/data/zzy/drivelm/reproduction/qwen_vl_v038

while tmux has-session -t v038_anchor_candidates 2>/dev/null \
   || tmux has-session -t v038_tag3_remaining 2>/dev/null; do
  sleep 15
done

for shard in 0 1 2; do
  test -s "$WORK/anchor_candidates/v037b_anchor_shard${shard}.jsonl.report.json"
done
test -s "$WORK/tag3_candidates/v037b_tag3.jsonl"
for shard in 0 1 2; do
  test -s "$WORK/tag3_candidates/v037b_tag3_remaining_shard${shard}.jsonl.report.json"
done

"$PY" "$ROOT/reproduction/qwen_vl_v038/build_grounding_preferences.py" \
  --train-jsonl /mnt/data/zzy/drivelm/reproduction/qwen_vl_stage2_ablation_v1/qwen_v033_c00_train.jsonl \
  --dev-jsonl /mnt/data/zzy/drivelm/reproduction/qwen_vl_stage2_ablation_v1/qwen_v033_c00_dev.jsonl \
  --anchor-candidate-jsonl "$WORK/anchor_candidates/v037b_anchor_shard0.jsonl" \
  --anchor-candidate-jsonl "$WORK/anchor_candidates/v037b_anchor_shard1.jsonl" \
  --anchor-candidate-jsonl "$WORK/anchor_candidates/v037b_anchor_shard2.jsonl" \
  --tag3-candidate-jsonl "$WORK/tag3_candidates/v037b_tag3.jsonl" \
  --tag3-candidate-jsonl "$WORK/tag3_candidates/v037b_tag3_remaining_shard0.jsonl" \
  --tag3-candidate-jsonl "$WORK/tag3_candidates/v037b_tag3_remaining_shard1.jsonl" \
  --tag3-candidate-jsonl "$WORK/tag3_candidates/v037b_tag3_remaining_shard2.jsonl" \
  --replay-jsonl /mnt/data/zzy/drivelm/reproduction/qwen_vl_v037b/balanced_preferences.jsonl \
  --output-jsonl "$WORK/grounding_balanced_preferences.jsonl" \
  --per-task 1026 --max-coordinate-f1 0.75 \
  --min-length-ratio 0.60 --max-length-ratio 1.45 --seed 42

echo V038_GROUNDING_PREFERENCES_COMPLETE
