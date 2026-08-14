#!/usr/bin/env bash
set -euo pipefail
PY=/home/zhangzongyuan/anaconda3/envs/radargym-rl/bin/python
TRAIN=/home/zhangzongyuan/Myproject/drivelm/DriveLM-main/reproduction/qwen_vl_v039_mol/train_expert_ddp.py
BASE=/mnt/data/zzy/drivelm/models/Qwen2.5-VL-7B-Instruct
START=/mnt/data/zzy/drivelm/models/qwen2.5-vl-7b-drivelm-v037b-anchored-dpo-seed42/checkpoint-75
DATA=/mnt/data/zzy/drivelm/reproduction/qwen_vl_v039_mol/manifests
MODELS=/mnt/data/zzy/drivelm/models/qwen2.5-vl-7b-drivelm-v039a-mol-pilot
LOGS=/mnt/data/zzy/drivelm/reproduction/qwen_vl_v039_mol/logs
mkdir -p "$MODELS" "$LOGS"
tasks=(perception prediction planning behavior)
gpus=(0 2 3 1)
pids=()
for i in 0 1 2 3; do
  task=${tasks[$i]}; gpu=${gpus[$i]}
  CUDA_VISIBLE_DEVICES=$gpu "$PY" "$TRAIN" \
    --experiment-name "v039a-mol-${task}-pilot" \
    --model-path "$BASE" --adapter-path "$START" \
    --train-jsonl "$DATA/train_${task}.jsonl" \
    --output-dir "$MODELS/expert_${task}" \
    --per-device-batch-size 1 --gradient-accumulation-steps 2 \
    --learning-rate 2e-6 --max-steps 100 --save-steps 0 \
    --max-pixels 100352 --max-length 4096 --seed 42 \
    > "$LOGS/train_${task}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if (( failed )); then echo V039A_EXPERT_TRAIN_FAILED; exit 1; fi
for task in "${tasks[@]}"; do test -s "$MODELS/expert_${task}/training_report.json"; done
echo V039A_EXPERT_TRAIN_COMPLETE
