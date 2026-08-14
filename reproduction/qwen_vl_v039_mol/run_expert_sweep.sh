#!/usr/bin/env bash
set -euo pipefail

# v0.39B: a clean, longer-horizon rerun from the frozen v0.37B start.
# v0.39A is preserved byte-for-byte as the 100-update feasibility result.
PY=/home/zhangzongyuan/anaconda3/envs/radargym-rl/bin/python
TRAIN=/home/zhangzongyuan/Myproject/drivelm/DriveLM-main/reproduction/qwen_vl_v039_mol/train_expert_ddp.py
BASE=/mnt/data/zzy/drivelm/models/Qwen2.5-VL-7B-Instruct
START=/mnt/data/zzy/drivelm/models/qwen2.5-vl-7b-drivelm-v037b-anchored-dpo-seed42/checkpoint-75
DATA=/mnt/data/zzy/drivelm/reproduction/qwen_vl_v039_mol/manifests
MODELS=/mnt/data/zzy/drivelm/models/qwen2.5-vl-7b-drivelm-v039b-mol-sweep
LOGS=/mnt/data/zzy/drivelm/reproduction/qwen_vl_v039_mol/sweep_logs

mkdir -p "$MODELS" "$LOGS"
tasks=(perception prediction planning behavior)
gpus=(0 2 3 1)
pids=()

for i in 0 1 2 3; do
  task=${tasks[$i]}
  gpu=${gpus[$i]}
  output="$MODELS/expert_${task}"
  if [[ -s "$output/training_report.json" ]]; then
    echo "skip completed expert: $task"
    continue
  fi
  CUDA_VISIBLE_DEVICES=$gpu "$PY" "$TRAIN" \
    --experiment-name "v039b-mol-${task}-sweep" \
    --model-path "$BASE" \
    --adapter-path "$START" \
    --train-jsonl "$DATA/train_${task}.jsonl" \
    --output-dir "$output" \
    --per-device-batch-size 1 \
    --gradient-accumulation-steps 2 \
    --learning-rate 2e-6 \
    --max-steps 500 \
    --save-steps 100 \
    --max-pixels 100352 \
    --max-length 4096 \
    --seed 42 \
    > "$LOGS/train_${task}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
if (( failed )); then
  echo V039B_EXPERT_SWEEP_FAILED
  exit 1
fi

for task in "${tasks[@]}"; do
  test -s "$MODELS/expert_${task}/training_report.json"
  for step in 100 200 300 400 500; do
    test -s "$MODELS/expert_${task}/checkpoint-${step}/adapter_model.safetensors"
  done
done
echo V039B_EXPERT_SWEEP_COMPLETE
