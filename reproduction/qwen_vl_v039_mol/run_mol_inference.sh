#!/usr/bin/env bash
set -euo pipefail
PY=/home/zhangzongyuan/anaconda3/envs/radargym-rl/bin/python
ROOT=/home/zhangzongyuan/Myproject/drivelm/DriveLM-main
INFER=$ROOT/reproduction/qwen_vl/infer.py
BASE=/mnt/data/zzy/drivelm/models/Qwen2.5-VL-7B-Instruct
MODELS=/mnt/data/zzy/drivelm/models/qwen2.5-vl-7b-drivelm-v039a-mol-pilot
DATA=/mnt/data/zzy/drivelm/reproduction/qwen_vl_v039_mol/manifests
OUT=/mnt/data/zzy/drivelm/reproduction/qwen_vl_v039_mol/mol_predictions
LOG=/mnt/data/zzy/drivelm/reproduction/qwen_vl_v039_mol/logs
mkdir -p "$OUT" "$LOG"
run_task() {
 local gpu=$1 task=$2
 CUDA_VISIBLE_DEVICES=$gpu "$PY" "$INFER" --model-path "$BASE" --adapter-path "$MODELS/expert_${task}" --input-jsonl "$DATA/dev_${task}.jsonl" --output-json "$OUT/${task}_predictions.json" --device cuda:0 --dtype bf16 --max-new-tokens 256 --batch-size 24 --min-pixels 25088 --max-pixels 100352 --resume > "$LOG/infer_${task}.log" 2>&1
}
run_task 2 planning & p1=$!
run_task 3 perception & p2=$!
( run_task 1 prediction; run_task 1 behavior ) & p3=$!
wait "$p1"; wait "$p2"; wait "$p3"
"$PY" "$ROOT/reproduction/qwen_vl_v039_mol/merge_task_predictions.py" --references-jsonl /mnt/data/zzy/drivelm/reproduction/qwen_vl_stage2_ablation_v1/qwen_v033_c00_dev.jsonl --prediction-dir "$OUT" --output-json "$OUT/mol_dev_predictions.json" --report-json "$OUT/merge_report.json"
echo V039A_MOL_INFERENCE_COMPLETE
