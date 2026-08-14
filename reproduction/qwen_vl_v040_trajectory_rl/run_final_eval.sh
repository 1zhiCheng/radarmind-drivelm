#!/usr/bin/env bash
set -euo pipefail

PY=/home/zhangzongyuan/anaconda3/envs/radargym-rl/bin/python
ROOT=/home/zhangzongyuan/Myproject/drivelm/DriveLM-main
HERE=$ROOT/reproduction/qwen_vl_v040_trajectory_rl
RUN=/mnt/data/zzy/drivelm/reproduction/qwen_vl_v040_trajectory_rl
BASE=/mnt/data/zzy/drivelm/models/Qwen2.5-VL-7B-Instruct
DEV=$RUN/dataset/dev.parquet
OUT=$RUN/final_eval
mkdir -p "$OUT/logs" "$OUT/predictions" "$OUT/metrics"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

"$PY" "$HERE/summarize_training.py" \
  --grpo-log "$RUN/logs/train_grpo.log" --gspo-log "$RUN/logs/train_gspo.log" \
  --output-json "$OUT/frozen_selection.json"
"$PY" "$HERE/build_eval_references.py" --dev-parquet "$DEV" \
  --source-jsonl /mnt/data/zzy/drivelm/reproduction/qwen_vl_stage2_ablation_v1/qwen_v033_c00_dev.jsonl \
  --output-jsonl "$OUT/planning_references.jsonl"

run_one() {
  local gpu=$1 name=$2 adapter=$3
  local predictions="$OUT/predictions/${name}.json"
  [[ -s "$predictions" ]] && return
  CUDA_VISIBLE_DEVICES=$gpu "$PY" "$HERE/infer_trajectory.py" \
    --model-path "$BASE" --adapter-path "$adapter" --dev-parquet "$DEV" \
    --output-json "$predictions" --device cuda:0 --batch-size 12 \
    --max-new-tokens 256 --min-pixels 25088 --max-pixels 100352 --resume \
    > "$OUT/logs/infer_${name}.log" 2>&1
}

run_one 0 baseline /mnt/data/zzy/drivelm/models/qwen2.5-vl-7b-drivelm-v039b-mol-sweep/expert_planning/checkpoint-700 & p0=$!
run_one 2 grpo70 "$RUN/exports/grpo-step70/lora_adapter" & p1=$!
run_one 3 gspo90 "$RUN/exports/gspo-step90/lora_adapter" & p2=$!
failed=0
for pid in "$p0" "$p1" "$p2"; do wait "$pid" || failed=1; done
(( failed == 0 )) || exit 1

for name in baseline grpo70 gspo90; do
  "$PY" "$HERE/evaluate_trajectory.py" --dev-parquet "$DEV" \
    --predictions-json "$OUT/predictions/${name}.json" \
    --output-json "$OUT/metrics/${name}_trajectory.json"
  "$PY" "$ROOT/reproduction/qwen_vl/evaluate_offline.py" \
    --references-jsonl "$OUT/planning_references.jsonl" \
    --predictions-json "$OUT/predictions/${name}.json" \
    --output-json "$OUT/metrics/${name}_offline.json"
done
printf 'V040_OFFLINE_EVAL_COMPLETE\n' > "$OUT/OFFLINE_COMPLETE"
