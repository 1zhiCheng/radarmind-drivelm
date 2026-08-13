#!/usr/bin/env bash
set -euo pipefail

PY=/home/zhangzongyuan/anaconda3/envs/radargym-rl/bin/python
ROOT=/home/zhangzongyuan/Myproject/drivelm/DriveLM-main
DEV=/mnt/data/zzy/drivelm/reproduction/qwen_vl_stage2_ablation_v1/qwen_v033_c00_dev.jsonl
BASE_MODEL=/mnt/data/zzy/drivelm/models/Qwen2.5-VL-7B-Instruct
MODEL=/mnt/data/zzy/drivelm/models/qwen2.5-vl-7b-drivelm-v038a-grounding-dpo-seed42
BASE_OFFLINE=/mnt/data/zzy/drivelm/reproduction/qwen_vl_v037b/checkpoint_sweep/checkpoint-75-dev-metrics.json
BASE_STRUCT=/mnt/data/zzy/drivelm/reproduction/qwen_vl_v038/v037b_baseline_structural.json
BASE_PRED=/mnt/data/zzy/drivelm/reproduction/qwen_vl_v037b/checkpoint_sweep/checkpoint-75-dev-predictions.json
BASE_DS=/mnt/data/zzy/drivelm/reproduction/qwen_vl_v037b/checkpoint-75-drivelm-ds.json
OUT=/mnt/data/zzy/drivelm/reproduction/qwen_vl_v038/checkpoint_sweep
LOG=/mnt/data/zzy/drivelm/reproduction/qwen_vl_v038/logs
CACHE=/mnt/data/zzy/drivelm/reproduction/drivelm_ds/deepseek_judge.sqlite
GPUS=(GPU-9e49353e-f137-3d6d-f7d5-987593a56d30 GPU-07f03596-83e4-920f-9669-fbec78272f67 GPU-8f2fb421-991d-f5eb-c811-448f7e105297 GPU-da6b809f-e2bb-bfbe-7f17-8093dd289ed4)
STEPS=(25 50 75 100)

mkdir -p "$OUT" "$LOG"
run_ds() {
  local references=$1
  local predictions=$2
  local output=$3
  local attempt=1
  cd "$ROOT/reproduction/drivelm_ds_eval"
  until env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
    "$PY" evaluate.py --references-jsonl "$references" \
    --predictions-json "$predictions" --output-json "$output" \
    --cache-file "$CACHE" --workers 8; do
    if (( attempt >= 10 )); then
      echo "DriveLM-DS failed after $attempt attempts" >&2
      return 12
    fi
    attempt=$((attempt + 1))
    sleep 30
  done
  cd "$ROOT"
}

while tmux has-session -t v038_train_pipeline 2>/dev/null; do sleep 15; done
test -s "$MODEL/training_report.json"
cd "$ROOT"

pids=()
for index in 0 1 2 3; do
  step=${STEPS[$index]}
  (
    CUDA_VISIBLE_DEVICES=${GPUS[$index]} "$PY" reproduction/qwen_vl/infer.py \
      --model-path "$BASE_MODEL" --adapter-path "$MODEL/checkpoint-$step" \
      --input-jsonl "$DEV" --output-json "$OUT/checkpoint-$step-dev-predictions.json" \
      --device cuda:0 --max-new-tokens 256 --batch-size 16 \
      --max-pixels 100352 --resume
  ) > "$LOG/checkpoint-$step-infer.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

for step in "${STEPS[@]}"; do
  "$PY" reproduction/qwen_vl/evaluate_offline.py \
    --references-jsonl "$DEV" --predictions-json "$OUT/checkpoint-$step-dev-predictions.json" \
    --output-json "$OUT/checkpoint-$step-dev-metrics.json"
  "$PY" reproduction/qwen_vl_v038/evaluate_structural.py \
    --references-jsonl "$DEV" --predictions-json "$OUT/checkpoint-$step-dev-predictions.json" \
    --output-json "$OUT/checkpoint-$step-structural.json"
done

selected=$("$PY" reproduction/qwen_vl_v038/select_grounding_candidate.py \
  --baseline-offline "$BASE_OFFLINE" --baseline-structural "$BASE_STRUCT" \
  --sweep-dir "$OUT" --output-json "$OUT/offline_selection.json")
if [[ "$selected" == "NONE" ]]; then
  echo "No v0.38A checkpoint passed the frozen pre-judge gates."
  echo V038A_EVAL_COMPLETE_NO_CANDIDATE
  exit 0
fi

PRED="$OUT/checkpoint-$selected-dev-predictions.json"
DS="$OUT/checkpoint-$selected-drivelm-ds.json"
run_ds "$DEV" "$PRED" "$DS"
"$PY" reproduction/qwen_vl_v038/compare_grounding_results.py \
  --baseline-offline "$BASE_OFFLINE" \
  --candidate-offline "$OUT/checkpoint-$selected-dev-metrics.json" \
  --baseline-structural "$BASE_STRUCT" \
  --candidate-structural "$OUT/checkpoint-$selected-structural.json" \
  --baseline-drivelm-ds "$BASE_DS" --candidate-drivelm-ds "$DS" \
  --output-json "$OUT/v037b-vs-checkpoint-$selected-summary.json" \
  --output-markdown "$OUT/v037b-vs-checkpoint-$selected-summary.md"

COMMON="$OUT/common_gating_audit"
"$PY" reproduction/qwen_vl_v037b/build_common_gating_subset.py \
  --references-jsonl "$DEV" --baseline-predictions "$BASE_PRED" \
  --candidate-predictions "$PRED" --output-dir "$COMMON"
for name in baseline candidate; do
  "$PY" reproduction/qwen_vl/evaluate_offline.py \
    --references-jsonl "$COMMON/common_references.jsonl" \
    --predictions-json "$COMMON/${name}_predictions.json" \
    --output-json "$COMMON/${name}_offline.json"
  run_ds "$COMMON/common_references.jsonl" \
    "$COMMON/${name}_predictions.json" "$COMMON/${name}_drivelm_ds.json"
done
"$PY" reproduction/qwen_vl_v037b/summarize_common_gating.py \
  --subset-report "$COMMON/common_subset_report.json" \
  --baseline-offline "$COMMON/baseline_offline.json" \
  --candidate-offline "$COMMON/candidate_offline.json" \
  --baseline-drivelm-ds "$COMMON/baseline_drivelm_ds.json" \
  --candidate-drivelm-ds "$COMMON/candidate_drivelm_ds.json" \
  --output-json "$COMMON/common_gating_comparison.json"

echo V038A_EVAL_COMPLETE_SELECTED_$selected
