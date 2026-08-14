#!/usr/bin/env bash
set -euo pipefail

PY=/home/zhangzongyuan/anaconda3/envs/radargym-rl/bin/python
ROOT=/home/zhangzongyuan/Myproject/drivelm/DriveLM-main
HERE=$ROOT/reproduction/qwen_vl_v040_trajectory_rl
INFER=$ROOT/reproduction/qwen_vl/infer.py
DEV=/mnt/data/zzy/drivelm/reproduction/qwen_vl_stage2_ablation_v1/qwen_v033_c00_dev.jsonl
DATA=/mnt/data/zzy/drivelm/reproduction/qwen_vl_v039_mol/manifests
BASE=/mnt/data/zzy/drivelm/models/Qwen2.5-VL-7B-Instruct
MOL=/mnt/data/zzy/drivelm/reproduction/qwen_vl_v039_mol/sweep/checkpoint-700/mol_dev_predictions.json
RLP=/mnt/data/zzy/drivelm/reproduction/qwen_vl_v040_trajectory_rl/final_eval/predictions
OUT=/mnt/data/zzy/drivelm/reproduction/qwen_vl_v040_trajectory_rl/full_system
mkdir -p "$OUT/raw_tasks" "$OUT/predictions" "$OUT/metrics" "$OUT/logs" "$OUT/same_id"

for name in grpo70 gspo90; do
  "$PY" "$HERE/build_full_system_predictions.py" --references-jsonl "$DEV" --mol-predictions "$MOL" --planning-predictions "$RLP/$name.json" --output-json "$OUT/predictions/$name.json" --report-json "$OUT/predictions/$name"_merge_report.json
done

run_raw() {
  local gpu=$1
  local task=$2
  local batch=$3
  local final="$OUT/raw_tasks/$task"_predictions.json
  if [[ -s "$final" ]]; then return; fi
  CUDA_VISIBLE_DEVICES=$gpu CUDA_DEVICE_ORDER=PCI_BUS_ID "$PY" "$INFER" --model-path "$BASE" --input-jsonl "$DATA/dev_$task.jsonl" --output-json "$final" --device cuda:0 --dtype bf16 --max-new-tokens 256 --batch-size "$batch" --min-pixels 25088 --max-pixels 100352 --resume > "$OUT/logs/raw_$task.log" 2>&1
}

run_raw 0 planning 20 & p0=$!
run_raw 2 perception 20 & p2=$!
run_raw 3 prediction 20 & p3=$!
run_raw 1 behavior 20 & p1=$!
failed=0
for pid in "$p0" "$p2" "$p3" "$p1"; do
  wait "$pid" || failed=1
done
if (( failed != 0 )); then
  echo "raw zero-shot inference failed" >&2
  exit 1
fi

exec "$HERE/run_full_system_post_eval.sh"

