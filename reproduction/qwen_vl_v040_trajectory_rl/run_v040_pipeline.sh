#!/usr/bin/env bash
set -euo pipefail

PY=/home/zhangzongyuan/anaconda3/envs/radargym-rl/bin/python
ROOT=/home/zhangzongyuan/Myproject/drivelm/DriveLM-main
HERE=$ROOT/reproduction/qwen_vl_v040_trajectory_rl
RUN=/mnt/data/zzy/drivelm/reproduction/qwen_vl_v040_trajectory_rl
UP=$RUN/upstream_train_predictions
DATA=$RUN/dataset
DEV_PRED=/mnt/data/zzy/drivelm/reproduction/qwen_vl_v039_mol/sweep/checkpoint-700
MODEL=/mnt/data/zzy/drivelm/models/Qwen2.5-VL-7B-Instruct
mkdir -p "$RUN/logs" "$DATA"

while [[ ! -s "$UP/COMPLETE" ]]; do sleep 20; done
"$PY" "$HERE/build_trajectory_dataset.py" \
  --train-jsonl /mnt/data/zzy/drivelm/reproduction/qwen_vl_v039_mol/manifests/train_shared_round_robin.jsonl \
  --dev-jsonl /mnt/data/zzy/drivelm/reproduction/qwen_vl_stage2_ablation_v1/qwen_v033_c00_dev.jsonl \
  --train-predictions "$UP/perception_predictions.json" "$UP/prediction_predictions.json" \
  --dev-predictions "$DEV_PRED/perception_predictions.json" "$DEV_PRED/prediction_predictions.json" \
  --output-dir "$DATA" > "$RUN/logs/build_dataset.log" 2>&1
"$PY" "$HERE/validate_trajectory_dataset.py" --model-path "$MODEL" \
  --train-parquet "$DATA/train.parquet" --dev-parquet "$DATA/dev.parquet" \
  --report-json "$DATA/loader_validation.json" > "$RUN/logs/validate_dataset.log" 2>&1
cd "$HERE"; "$PY" -m pytest -q test_trajectory_reward.py > "$RUN/logs/reward_tests.log" 2>&1

# Matched one-update smoke runs must both pass before either full run starts.
# Keep the marker so a pipeline restart never repeats hours of completed work.
if [[ ! -s "$RUN/SMOKE_COMPLETE" ]]; then
  "$HERE/run_verl_rl.sh" grpo trainer.total_training_steps=1 trainer.total_epochs=1 \
    data.train_max_samples=12 data.val_max_samples=12 \
    trainer.save_freq=1 trainer.test_freq=1 trainer.resume_mode=disable \
    trainer.default_local_dir=/mnt/data/zzy/drivelm/models/qwen2.5-vl-7b-drivelm-v040-smoke-grpo \
    > "$RUN/logs/smoke_grpo.log" 2>&1
  "$HERE/run_verl_rl.sh" gspo trainer.total_training_steps=1 trainer.total_epochs=1 \
    data.train_max_samples=12 data.val_max_samples=12 \
    trainer.save_freq=1 trainer.test_freq=1 trainer.resume_mode=disable \
    trainer.default_local_dir=/mnt/data/zzy/drivelm/models/qwen2.5-vl-7b-drivelm-v040-smoke-gspo \
    > "$RUN/logs/smoke_gspo.log" 2>&1
  printf 'V040_SMOKE_COMPLETE\n' > "$RUN/SMOKE_COMPLETE"
fi

# Same initialization and budget; the loss mode is the sole algorithm change.
if [[ ! -s "$RUN/GRPO_COMPLETE" ]]; then
  "$HERE/run_verl_rl.sh" grpo trainer.resume_mode=auto > "$RUN/logs/train_grpo.log" 2>&1
  printf 'V040_GRPO_COMPLETE\n' > "$RUN/GRPO_COMPLETE"
fi
if [[ ! -s "$RUN/GSPO_COMPLETE" ]]; then
  "$HERE/run_verl_rl.sh" gspo trainer.resume_mode=auto > "$RUN/logs/train_gspo.log" 2>&1
  printf 'V040_GSPO_COMPLETE\n' > "$RUN/GSPO_COMPLETE"
fi
printf 'V040_TRAINING_COMPLETE\n' > "$RUN/TRAINING_COMPLETE"
