#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zhangzongyuan/Myproject/drivelm/DriveLM-main
PY=/home/zhangzongyuan/anaconda3/envs/radargym-rl/bin/python
ACCELERATE=/home/zhangzongyuan/anaconda3/envs/radargym-rl/bin/accelerate
BASE=/mnt/data/zzy/drivelm/models/Qwen2.5-VL-7B-Instruct
B10=/mnt/data/zzy/drivelm/models/qwen2.5-vl-7b-drivelm-v036-b10-seed42
TRAIN=/mnt/data/zzy/drivelm/reproduction/qwen_vl_stage2_ablation_v1/qwen_v033_c00_train.jsonl
DEV=/mnt/data/zzy/drivelm/reproduction/qwen_vl_stage2_ablation_v1/qwen_v033_c00_dev.jsonl
WORK=/mnt/data/zzy/drivelm/reproduction/qwen_vl_v037a
MODEL_OUT=/mnt/data/zzy/drivelm/models/qwen2.5-vl-7b-drivelm-v037a-b10-dpo-seed42
CACHE=/mnt/data/zzy/drivelm/reproduction/drivelm_ds/deepseek_judge.sqlite
GPU0=GPU-9e49353e-f137-3d6d-f7d5-987593a56d30
GPU1=GPU-07f03596-83e4-920f-9669-fbec78272f67
GPU2=GPU-8f2fb421-991d-f5eb-c811-448f7e105297
TARGETS=(8699 8698 8698)
GPUS=("$GPU0" "$GPU1" "$GPU2")

mkdir -p "$WORK"/{candidates,reference,dev_shards,logs,outputs}
cd "$ROOT"

count_lines() {
  local path=$1
  if [[ -f "$path" ]]; then
    wc -l < "$path"
  else
    echo 0
  fi
}

start_candidate_fallback() {
  local shard=$1
  local session="v037a_cand${shard}"
  local output="$WORK/candidates/b10_train_shard${shard}.jsonl"
  if tmux has-session -t "$session" 2>/dev/null; then
    return
  fi
  tmux new-session -d -s "$session" \
    "CUDA_VISIBLE_DEVICES=${GPUS[$shard]} $PY reproduction/qwen_vl_v037a/generate_candidates.py --model-path $BASE --adapter-path $B10 --train-jsonl $TRAIN --output-jsonl $output --num-shards 3 --shard-index $shard --batch-size 8 --candidates-per-record 2 --max-new-tokens 128 --temperature 0.8 --top-p 0.9 --repetition-penalty 1.05 --seed 42 --resume 2>&1 | tee -a $WORK/logs/candidates_shard${shard}.log"
}

echo "[$(date --iso-8601=seconds)] waiting for complete candidate coverage"
while true; do
  complete=1
  status=()
  for shard in 0 1 2; do
    output="$WORK/candidates/b10_train_shard${shard}.jsonl"
    count=$(count_lines "$output")
    target=${TARGETS[$shard]}
    status+=("$count/$target")
    if (( count < target )); then
      complete=0
      start_candidate_fallback "$shard"
    elif (( count > target )); then
      echo "candidate shard $shard exceeds target: $count > $target" >&2
      exit 10
    fi
  done
  echo "[$(date --iso-8601=seconds)] candidates ${status[*]}"
  (( complete == 1 )) && break
  sleep 30
done

PREFERENCES="$WORK/preferences_b10_train.jsonl"
"$PY" reproduction/qwen_vl_v037a/build_preferences.py \
  --train-jsonl "$TRAIN" \
  --dev-jsonl "$DEV" \
  --candidate-jsonl "$WORK/candidates/b10_train_shard0.jsonl" \
  --candidate-jsonl "$WORK/candidates/b10_train_shard1.jsonl" \
  --candidate-jsonl "$WORK/candidates/b10_train_shard2.jsonl" \
  --output-jsonl "$PREFERENCES"
PAIR_COUNT=$(count_lines "$PREFERENCES")
"$PY" reproduction/qwen_vl_v037a/audit_preferences.py \
  --report-json "$PREFERENCES.report.json" \
  --min-pairs 500 --min-pairs-per-task 50
echo "[$(date --iso-8601=seconds)] built $PAIR_COUNT preference pairs"

WARM_REFERENCE="$WORK/reference/warmup_reference.jsonl"
skip_args=()
merge_warm_args=()
if tmux has-session -t v037a_ref_warmup 2>/dev/null; then
  echo "[$(date --iso-8601=seconds)] waiting for A6000 reference warmup"
  while tmux has-session -t v037a_ref_warmup 2>/dev/null; do
    sleep 15
  done
fi
if [[ -s "$WARM_REFERENCE.report.json" && -s "$WARM_REFERENCE" ]]; then
  skip_args=(--skip-reference-jsonl "$WARM_REFERENCE")
  merge_warm_args=(--reference-jsonl "$WARM_REFERENCE")
  echo "[$(date --iso-8601=seconds)] reusing $(count_lines "$WARM_REFERENCE") A6000 reference scores"
fi

reference_pids=()
for shard in 0 1 2; do
  CUDA_VISIBLE_DEVICES=${GPUS[$shard]} "$PY" \
    reproduction/qwen_vl_v037a/precompute_reference.py \
    --model-path "$BASE" \
    --adapter-path "$B10" \
    --preference-jsonl "$PREFERENCES" \
    --output-jsonl "$WORK/reference/reference_shard${shard}.jsonl" \
    --num-shards 3 --shard-index "$shard" \
    --normalization sum "${skip_args[@]}" --resume \
    > "$WORK/logs/reference_shard${shard}.log" 2>&1 &
  reference_pids+=("$!")
done
for pid in "${reference_pids[@]}"; do
  wait "$pid"
done

REFERENCE="$WORK/reference/preferences_with_b10_reference.jsonl"
"$PY" reproduction/qwen_vl_v037a/merge_reference_shards.py \
  --preference-jsonl "$PREFERENCES" \
  "${merge_warm_args[@]}" \
  --reference-jsonl "$WORK/reference/reference_shard0.jsonl" \
  --reference-jsonl "$WORK/reference/reference_shard1.jsonl" \
  --reference-jsonl "$WORK/reference/reference_shard2.jsonl" \
  --output-jsonl "$REFERENCE"

CUDA_VISIBLE_DEVICES="$GPU0,$GPU1,$GPU2" "$ACCELERATE" launch \
  --multi_gpu --num_processes 3 --mixed_precision bf16 \
  --main_process_port 29637 \
  reproduction/qwen_vl_v037a/train_dpo_ddp.py \
  --experiment-name v037a-b10-dpo-seed42 \
  --model-path "$BASE" \
  --adapter-path "$B10" \
  --reference-jsonl "$REFERENCE" \
  --output-dir "$MODEL_OUT" \
  --epochs 1 --gradient-accumulation-steps 4 \
  --learning-rate 5e-6 --beta 0.1 \
  --max-pixels 100352 --max-length 4096 --save-steps 100 \
  2>&1 | tee "$WORK/logs/dpo_train.log"

"$PY" reproduction/qwen_vl_v037a/shard_inference.py shard \
  --input-jsonl "$DEV" --output-dir "$WORK/dev_shards" --num-shards 3

infer_pids=()
for shard in 0 1 2; do
  (
    output="$WORK/outputs/dpo_dev_shard${shard}.json"
    if ! CUDA_VISIBLE_DEVICES=${GPUS[$shard]} "$PY" reproduction/qwen_vl/infer.py \
      --model-path "$BASE" --adapter-path "$MODEL_OUT" \
      --input-jsonl "$WORK/dev_shards/dev_shard${shard}.jsonl" \
      --output-json "$output" --device cuda:0 --max-new-tokens 256 \
      --batch-size 8 --max-pixels 100352 --resume; then
      CUDA_VISIBLE_DEVICES=${GPUS[$shard]} "$PY" reproduction/qwen_vl/infer.py \
        --model-path "$BASE" --adapter-path "$MODEL_OUT" \
        --input-jsonl "$WORK/dev_shards/dev_shard${shard}.jsonl" \
        --output-json "$output" --device cuda:0 --max-new-tokens 256 \
        --batch-size 4 --max-pixels 100352 --resume
    fi
  ) > "$WORK/logs/dev_infer_shard${shard}.log" 2>&1 &
  infer_pids+=("$!")
done
for pid in "${infer_pids[@]}"; do
  wait "$pid"
done

PREDICTIONS="$WORK/outputs/b10_dpo_dev_predictions.json"
"$PY" reproduction/qwen_vl_v037a/shard_inference.py merge \
  --source-jsonl "$DEV" \
  --prediction-json "$WORK/outputs/dpo_dev_shard0.json" \
  --prediction-json "$WORK/outputs/dpo_dev_shard1.json" \
  --prediction-json "$WORK/outputs/dpo_dev_shard2.json" \
  --output-json "$PREDICTIONS"

"$PY" reproduction/qwen_vl/evaluate_offline.py \
  --references-jsonl "$DEV" --predictions-json "$PREDICTIONS" \
  --output-json "$WORK/outputs/b10_dpo_dev_metrics.json"

cd "$ROOT/reproduction/drivelm_ds_eval"
attempt=1
until "$PY" evaluate.py \
  --references-jsonl "$DEV" --predictions-json "$PREDICTIONS" \
  --output-json "$WORK/outputs/b10_dpo_drivelm_ds.json" \
  --cache-file "$CACHE" --workers 8; do
  if (( attempt >= 10 )); then
    echo "DriveLM-DS failed after $attempt attempts" >&2
    exit 12
  fi
  attempt=$((attempt + 1))
  sleep 30
done

cd "$ROOT"
"$PY" reproduction/qwen_vl_stage1/compare.py \
  --references-jsonl "$DEV" \
  --baseline-predictions /mnt/data/zzy/drivelm/reproduction/qwen_vl_v036/b10_dev_predictions.json \
  --enhanced-predictions "$PREDICTIONS" \
  --output-json "$WORK/outputs/paired_b10_vs_b10_dpo.json" \
  --output-markdown "$WORK/outputs/paired_b10_vs_b10_dpo.md"

"$PY" reproduction/qwen_vl_v037a/compare_results.py \
  --baseline-offline /mnt/data/zzy/drivelm/reproduction/qwen_vl_v036/b10_dev_metrics.json \
  --candidate-offline "$WORK/outputs/b10_dpo_dev_metrics.json" \
  --baseline-drivelm-ds /mnt/data/zzy/drivelm/reproduction/qwen_vl_v036/b10_drivelm_ds.json \
  --candidate-drivelm-ds "$WORK/outputs/b10_dpo_drivelm_ds.json" \
  --output-json "$WORK/outputs/b10_vs_b10_dpo_summary.json" \
  --output-markdown "$WORK/outputs/b10_vs_b10_dpo_summary.md"

echo "[$(date --iso-8601=seconds)] v0.37A full pipeline complete"
