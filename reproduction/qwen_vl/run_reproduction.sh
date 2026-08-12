#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}
DATA_ROOT=${DATA_ROOT:-$PWD}
MODEL_PATH=${MODEL_PATH:-$PWD/models/Qwen2.5-VL-3B-Instruct}
GPU_ID=${GPU_ID:-0}
OUTPUT_ROOT=${OUTPUT_ROOT:-${DATA_ROOT}/data/reproduction/qwen_vl}
ADAPTER_DIR=${ADAPTER_DIR:-${DATA_ROOT}/models/qwen2.5-vl-3b-drivelm-sixcam}

"${PYTHON_BIN}" reproduction/qwen_vl/build_dataset.py \
  --train-json "${DATA_ROOT}/data/QA_dataset_nus/v1_1_train_nus.json" \
  --val-json "${DATA_ROOT}/data/QA_dataset_nus/v1_1_val_nus_q_only.json" \
  --output-dir "${OUTPUT_ROOT}" --seed 42 --dev-ratio 0.1

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU_ID} "${PYTHON_BIN}" reproduction/qwen_vl/train.py \
  --model-path "${MODEL_PATH}" --train-jsonl "${OUTPUT_ROOT}/qwen_train.jsonl" \
  --output-dir "${ADAPTER_DIR}" --device cuda:0 --batch-size 4 \
  --gradient-accumulation-steps 1 --epochs 1 --max-steps 0 --save-steps 500

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU_ID} "${PYTHON_BIN}" reproduction/qwen_vl/infer.py \
  --model-path "${MODEL_PATH}" --adapter-path "${ADAPTER_DIR}" \
  --input-jsonl "${OUTPUT_ROOT}/qwen_dev.jsonl" \
  --output-json "${OUTPUT_ROOT}/dev_predictions.json" --device cuda:0 \
  --max-new-tokens 256 --resume

"${PYTHON_BIN}" reproduction/qwen_vl/evaluate_offline.py \
  --references-jsonl "${OUTPUT_ROOT}/qwen_dev.jsonl" \
  --predictions-json "${OUTPUT_ROOT}/dev_predictions.json" \
  --output-json "${OUTPUT_ROOT}/dev_metrics.json"
