#!/usr/bin/env bash
set -euo pipefail

if (( $# != 2 )); then
  echo "usage: $0 FROM_STEP TO_STEP" >&2
  exit 2
fi
FROM_STEP=$1
TO_STEP=$2
(( TO_STEP > FROM_STEP )) || { echo "TO_STEP must be greater than FROM_STEP" >&2; exit 2; }

PY=/home/zhangzongyuan/anaconda3/envs/radargym-rl/bin/python
TRAIN=/home/zhangzongyuan/Myproject/drivelm/DriveLM-main/reproduction/qwen_vl_v039_mol/train_expert_ddp.py
BASE=/mnt/data/zzy/drivelm/models/Qwen2.5-VL-7B-Instruct
START=/mnt/data/zzy/drivelm/models/qwen2.5-vl-7b-drivelm-v037b-anchored-dpo-seed42/checkpoint-75
DATA=/mnt/data/zzy/drivelm/reproduction/qwen_vl_v039_mol/manifests
MODELS=/mnt/data/zzy/drivelm/models/qwen2.5-vl-7b-drivelm-v039b-mol-sweep
LOGS=/mnt/data/zzy/drivelm/reproduction/qwen_vl_v039_mol/sweep_logs
tasks=(perception prediction planning behavior)
gpus=(0 2 3 1)
pids=()

for i in 0 1 2 3; do
  task=${tasks[$i]}; gpu=${gpus[$i]}
  output="$MODELS/expert_${task}"
  resume="$output/checkpoint-${FROM_STEP}"
  test -s "$resume/adapter_model.safetensors"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" "$TRAIN" \
    --experiment-name "v039b-mol-${task}-${FROM_STEP}-to-${TO_STEP}" \
    --model-path "$BASE" --adapter-path "$START" \
    --resume-from-checkpoint "$resume" \
    --train-jsonl "$DATA/train_${task}.jsonl" --output-dir "$output" \
    --per-device-batch-size 1 --gradient-accumulation-steps 2 \
    --learning-rate 2e-6 --max-steps "$TO_STEP" --save-steps 100 \
    --max-pixels 100352 --max-length 4096 --seed 42 \
    > "$LOGS/continue_${task}_${FROM_STEP}_${TO_STEP}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
(( failed == 0 )) || { echo V039B_CONTINUATION_FAILED; exit 1; }
for task in "${tasks[@]}"; do
  for (( step=FROM_STEP+100; step<=TO_STEP; step+=100 )); do
    test -s "$MODELS/expert_${task}/checkpoint-${step}/adapter_model.safetensors"
  done
done
echo "V039B_CONTINUATION_COMPLETE from=$FROM_STEP to=$TO_STEP"
