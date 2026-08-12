#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zhangzongyuan/Myproject/drivelm/DriveLM-main
PY=/home/zhangzongyuan/anaconda3/envs/radargym-rl/bin/python
BASE=/mnt/data/zzy/drivelm/models/Qwen2.5-VL-7B-Instruct
ADAPTER=/mnt/data/zzy/drivelm/models/qwen2.5-vl-7b-drivelm-v036-b11-seed42
REF=/mnt/data/zzy/drivelm/reproduction/qwen_vl_stage2_ablation_v1/qwen_v033_c00_dev.jsonl
OUT=/mnt/data/zzy/drivelm/reproduction/qwen_vl_v036
CACHE=/mnt/data/zzy/drivelm/reproduction/drivelm_ds/deepseek_judge.sqlite
A6000_UUID=GPU-da6b809f-e2bb-bfbe-7f17-8093dd289ed4

while [[ ! -s "$ADAPTER/training_report.json" ]]; do
  sleep 30
done

for required in adapter_config.json adapter_model.safetensors training_report.json; do
  [[ -s "$ADAPTER/$required" ]] || {
    echo "missing post-training artifact: $ADAPTER/$required" >&2
    exit 2
  }
done

cd "$ROOT"
CUDA_VISIBLE_DEVICES="$A6000_UUID" "$PY" reproduction/qwen_vl/infer.py \
  --model-path "$BASE" \
  --adapter-path "$ADAPTER" \
  --input-jsonl "$REF" \
  --output-json "$OUT/b11_dev_predictions.json" \
  --device cuda:0 \
  --max-new-tokens 256 \
  --batch-size 1 \
  --max-pixels 200704 \
  --resume

"$PY" reproduction/qwen_vl/evaluate_offline.py \
  --references-jsonl "$REF" \
  --predictions-json "$OUT/b11_dev_predictions.json" \
  --output-json "$OUT/b11_dev_metrics.json"

cd "$ROOT/reproduction/drivelm_ds_eval"
until "$PY" evaluate.py \
  --references-jsonl "$REF" \
  --predictions-json "$OUT/b11_dev_predictions.json" \
  --output-json "$OUT/b11_drivelm_ds.json" \
  --cache-file "$CACHE" \
  --workers 8; do
  echo "DriveLM-DS failed; retrying in 30 seconds" >&2
  sleep 30
done

cd "$ROOT"
"$PY" reproduction/qwen_vl_stage1/compare.py \
  --references-jsonl "$REF" \
  --baseline-predictions /mnt/data/zzy/drivelm/reproduction/qwen_vl_stage2_ablation_v1/c00_dev_predictions.json \
  --enhanced-predictions "$OUT/b11_dev_predictions.json" \
  --output-json "$OUT/paired_c00ce_vs_b11.json" \
  --output-markdown "$OUT/paired_c00ce_vs_b11.md"

"$PY" reproduction/qwen_vl_stage1/compare.py \
  --references-jsonl "$REF" \
  --baseline-predictions "$OUT/b10_dev_predictions.json" \
  --enhanced-predictions "$OUT/b11_dev_predictions.json" \
  --output-json "$OUT/paired_b10_vs_b11.json" \
  --output-markdown "$OUT/paired_b10_vs_b11.md"

echo "B11 post-training pipeline complete"
