#!/usr/bin/env bash
set -euo pipefail

PY=/home/zhangzongyuan/anaconda3/envs/radargym-rl/bin/python
ROOT=/home/zhangzongyuan/Myproject/drivelm/DriveLM-main
INFER=$ROOT/reproduction/qwen_vl/infer.py
MERGE=$ROOT/reproduction/qwen_vl_v039_mol/merge_task_predictions.py
BASE=/mnt/data/zzy/drivelm/models/Qwen2.5-VL-7B-Instruct
MODELS=/mnt/data/zzy/drivelm/models/qwen2.5-vl-7b-drivelm-v039b-mol-sweep
DATA=/mnt/data/zzy/drivelm/reproduction/qwen_vl_v039_mol/manifests
DEV=/mnt/data/zzy/drivelm/reproduction/qwen_vl_stage2_ablation_v1/qwen_v033_c00_dev.jsonl
DIR=/mnt/data/zzy/drivelm/reproduction/qwen_vl_v039_mol/sweep
BASE_OFF=/mnt/data/zzy/drivelm/reproduction/qwen_vl_v039_mol/evaluation/mol_offline.json
BASE_ST=/mnt/data/zzy/drivelm/reproduction/qwen_vl_v039_mol/evaluation/mol_structural.json

mkdir -p "$DIR/logs" "$DIR/evaluation"

run_one() {
  local gpu=$1 task=$2 step=$3 out=$4
  local final="$out/${task}_predictions.json"
  if [[ -s "$final" ]]; then
    return
  fi
  CUDA_VISIBLE_DEVICES=$gpu "$PY" "$INFER" \
    --model-path "$BASE" \
    --adapter-path "$MODELS/expert_${task}/checkpoint-${step}" \
    --input-jsonl "$DATA/dev_${task}.jsonl" \
    --output-json "$final" \
    --device cuda:0 --dtype bf16 \
    --max-new-tokens 256 --batch-size 20 \
    --min-pixels 25088 --max-pixels 100352 --resume \
    > "$DIR/logs/checkpoint-${step}_${task}.log" 2>&1
}

for step in 100 200 300 400 500; do
  out="$DIR/checkpoint-${step}"
  mkdir -p "$out"
  run_one 0 perception "$step" "$out" & p0=$!
  run_one 2 prediction "$step" "$out" & p2=$!
  run_one 3 planning "$step" "$out" & p3=$!
  run_one 1 behavior "$step" "$out" & p1=$!
  failed=0
  for pid in "$p0" "$p2" "$p3" "$p1"; do wait "$pid" || failed=1; done
  (( failed == 0 )) || { echo "inference failed at checkpoint $step"; exit 1; }

  "$PY" "$MERGE" --references-jsonl "$DEV" --prediction-dir "$out" \
    --output-json "$out/mol_dev_predictions.json" --report-json "$out/merge_report.json"
  "$PY" "$ROOT/reproduction/qwen_vl/evaluate_offline.py" \
    --references-jsonl "$DEV" --predictions-json "$out/mol_dev_predictions.json" \
    --output-json "$DIR/evaluation/checkpoint-${step}_offline.json"
  "$PY" "$ROOT/reproduction/qwen_vl_v038/evaluate_structural.py" \
    --references-jsonl "$DEV" --predictions-json "$out/mol_dev_predictions.json" \
    --output-json "$DIR/evaluation/checkpoint-${step}_structural.json"
done

"$PY" "$ROOT/reproduction/qwen_vl_v039_mol/monitor_sweep.py" \
  --evaluation-dir "$DIR/evaluation" \
  --baseline-offline "$BASE_OFF" --baseline-structural "$BASE_ST" \
  --steps 100 200 300 400 500 --min-delta 0.0005 --patience 2 --max-step-cap 1500 \
  --output-json "$DIR/monitor_report.json"
echo V039B_SWEEP_EVALUATION_COMPLETE
