#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zhangzongyuan/Myproject/drivelm/DriveLM-main
PY=/home/zhangzongyuan/anaconda3/envs/radargym-rl/bin/python
DEV=/mnt/data/zzy/drivelm/reproduction/qwen_vl_stage2_ablation_v1/qwen_v033_c00_dev.jsonl
BASE_OFFLINE=/mnt/data/zzy/drivelm/reproduction/qwen_vl_v036/b10_dev_metrics.json
BASE_DS=/mnt/data/zzy/drivelm/reproduction/qwen_vl_v036/b10_drivelm_ds.json
SWEEP=/mnt/data/zzy/drivelm/reproduction/qwen_vl_v037b/checkpoint_sweep
OUT=/mnt/data/zzy/drivelm/reproduction/qwen_vl_v037b
CACHE=/mnt/data/zzy/drivelm/reproduction/drivelm_ds/deepseek_judge.sqlite

cd "$ROOT"
while tmux has-session -t v037b_eval_manager 2>/dev/null; do
  sleep 15
done

for step in 25 50 75 100; do
  test -s "$SWEEP/checkpoint-$step-dev-predictions.json"
  test -s "$SWEEP/checkpoint-$step-dev-metrics.json"
done

selected=$(
  "$PY" reproduction/qwen_vl_v037b/select_offline_candidate.py \
    --baseline "$BASE_OFFLINE" --sweep-dir "$SWEEP" \
    --output-json "$OUT/offline_selection.json"
)

if [[ "$selected" == "NONE" ]]; then
  echo "No checkpoint passed the pre-judge gate; DeepSeek was not called."
  echo V037B_POST_EVAL_COMPLETE
  exit 0
fi

PRED="$SWEEP/checkpoint-$selected-dev-predictions.json"
DS_OUT="$OUT/checkpoint-$selected-drivelm-ds.json"
cd "$ROOT/reproduction/drivelm_ds_eval"
attempt=1
until env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  "$PY" evaluate.py \
    --references-jsonl "$DEV" --predictions-json "$PRED" \
    --output-json "$DS_OUT" --cache-file "$CACHE" --workers 8; do
  if (( attempt >= 10 )); then
    echo "DriveLM-DS failed after $attempt attempts" >&2
    exit 12
  fi
  attempt=$((attempt + 1))
  sleep 30
done

cd "$ROOT"
"$PY" reproduction/qwen_vl_v037b/compare_results.py \
  --baseline-offline "$BASE_OFFLINE" \
  --candidate-offline "$SWEEP/checkpoint-$selected-dev-metrics.json" \
  --baseline-drivelm-ds "$BASE_DS" \
  --candidate-drivelm-ds "$DS_OUT" \
  --output-json "$OUT/b10_vs_checkpoint-$selected-summary.json" \
  --output-markdown "$OUT/b10_vs_checkpoint-$selected-summary.md"

echo V037B_POST_EVAL_COMPLETE
