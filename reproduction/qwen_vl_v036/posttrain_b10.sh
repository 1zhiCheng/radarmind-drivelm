#!/usr/bin/env bash
set -euo pipefail

# Required or portable defaults. Override every path from the shell when needed.
PYTHON_BIN=${PYTHON_BIN:-python}
BASE_MODEL=${BASE_MODEL:-$PWD/models/Qwen2.5-VL-7B-Instruct}
ADAPTER_DIR=${ADAPTER_DIR:-$PWD/models/radarmind-drivelm-b10}
DEV_JSONL=${DEV_JSONL:-$PWD/data/reproduction/qwen_vl/qwen_dev.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-$PWD/outputs/b10}
DEVICE=${DEVICE:-cuda:0}
DEEPSEEK_SECRET_FILE=${DEEPSEEK_SECRET_FILE:-$HOME/.config/radarmind/deepseek_api_key}

for required in adapter_config.json adapter_model.safetensors training_report.json; do
  test -s "$ADAPTER_DIR/$required" || {
    echo "Missing post-training artifact: $ADAPTER_DIR/$required" >&2
    exit 2
  }
done

mkdir -p "$OUTPUT_DIR"

"$PYTHON_BIN" reproduction/qwen_vl/infer.py \
  --model-path "$BASE_MODEL" \
  --adapter-path "$ADAPTER_DIR" \
  --input-jsonl "$DEV_JSONL" \
  --output-json "$OUTPUT_DIR/dev_predictions.json" \
  --device "$DEVICE" --max-new-tokens 256 --batch-size 1 --resume

"$PYTHON_BIN" reproduction/qwen_vl/evaluate_offline.py \
  --references-jsonl "$DEV_JSONL" \
  --predictions-json "$OUTPUT_DIR/dev_predictions.json" \
  --output-json "$OUTPUT_DIR/dev_metrics.json"

"$PYTHON_BIN" reproduction/drivelm_ds_eval/evaluate.py \
  --references-jsonl "$DEV_JSONL" \
  --predictions-json "$OUTPUT_DIR/dev_predictions.json" \
  --output-json "$OUTPUT_DIR/drivelm_ds.json" \
  --cache-file "$OUTPUT_DIR/deepseek_judge.sqlite" \
  --secret-file "$DEEPSEEK_SECRET_FILE"

echo "B10 inference and evaluation complete: $OUTPUT_DIR"
