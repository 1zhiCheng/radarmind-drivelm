#!/usr/bin/env bash
set -euo pipefail

PY=/home/zhangzongyuan/anaconda3/envs/radargym-rl/bin/python
ROOT=/home/zhangzongyuan/Myproject/drivelm/DriveLM-main
HERE=$ROOT/reproduction/qwen_vl_v040_trajectory_rl
DEV=/mnt/data/zzy/drivelm/reproduction/qwen_vl_stage2_ablation_v1/qwen_v033_c00_dev.jsonl
MOL=/mnt/data/zzy/drivelm/reproduction/qwen_vl_v039_mol/sweep/checkpoint-700/mol_dev_predictions.json
MOL_DS=/mnt/data/zzy/drivelm/reproduction/qwen_vl_v039_mol/sweep/evaluation/checkpoint-700_drivelm_ds.json
OUT=/mnt/data/zzy/drivelm/reproduction/qwen_vl_v040_trajectory_rl/full_system
CACHE=/mnt/data/zzy/drivelm/reproduction/drivelm_ds/deepseek_judge.sqlite

for task in perception prediction planning behavior; do
  while [[ ! -s "$OUT/raw_tasks/$task"_predictions.json ]]; do sleep 20; done
done

"$PY" "$ROOT/reproduction/qwen_vl_v039_mol/merge_task_predictions.py" --references-jsonl "$DEV" --prediction-dir "$OUT/raw_tasks" --output-json "$OUT/predictions/raw.json" --report-json "$OUT/predictions/raw_merge_report.json"

for name in raw grpo70 gspo90; do
  "$PY" "$ROOT/reproduction/qwen_vl/evaluate_offline.py" --references-jsonl "$DEV" --predictions-json "$OUT/predictions/$name.json" --output-json "$OUT/metrics/$name"_offline.json
  "$PY" "$ROOT/reproduction/qwen_vl_v038/evaluate_structural.py" --references-jsonl "$DEV" --predictions-json "$OUT/predictions/$name.json" --output-json "$OUT/metrics/$name"_structural.json
done
"$PY" "$ROOT/reproduction/qwen_vl/evaluate_offline.py" --references-jsonl "$DEV" --predictions-json "$MOL" --output-json "$OUT/metrics/mol700_offline.json"

for name in raw grpo70 gspo90; do
  cd "$ROOT/reproduction/drivelm_ds_eval"
  env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy "$PY" evaluate.py --references-jsonl "$DEV" --predictions-json "$OUT/predictions/$name.json" --output-json "$OUT/metrics/$name"_drivelm_ds.json --cache-file "$CACHE" --workers 8
done
cd "$ROOT"

for name in grpo70 gspo90; do
  PAIR=$OUT/same_id/$name
  "$PY" "$ROOT/reproduction/qwen_vl_v037b/build_common_gating_subset.py" --references-jsonl "$DEV" --baseline-predictions "$MOL" --candidate-predictions "$OUT/predictions/$name.json" --output-dir "$PAIR"
  for role in baseline candidate; do
    "$PY" "$ROOT/reproduction/qwen_vl/evaluate_offline.py" --references-jsonl "$PAIR/common_references.jsonl" --predictions-json "$PAIR/$role"_predictions.json --output-json "$PAIR/$role"_offline.json
    cd "$ROOT/reproduction/drivelm_ds_eval"
    env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy "$PY" evaluate.py --references-jsonl "$PAIR/common_references.jsonl" --predictions-json "$PAIR/$role"_predictions.json --output-json "$PAIR/$role"_drivelm_ds.json --cache-file "$CACHE" --workers 8
    cd "$ROOT"
  done
  "$PY" "$ROOT/reproduction/qwen_vl_v037b/summarize_common_gating.py" --subset-report "$PAIR/common_subset_report.json" --baseline-offline "$PAIR/baseline_offline.json" --candidate-offline "$PAIR/candidate_offline.json" --baseline-drivelm-ds "$PAIR/baseline_drivelm_ds.json" --candidate-drivelm-ds "$PAIR/candidate_drivelm_ds.json" --output-json "$PAIR/common_gating_comparison.json"
done

"$PY" "$HERE/summarize_full_system.py" --result-root "$OUT" --baseline-ds "$MOL_DS" --output-json "$OUT/final_summary.json" --output-markdown "$OUT/final_summary.md"
printf 'V040_FULL_SYSTEM_COMPLETE\n' > "$OUT/COMPLETE"
echo V040_FULL_SYSTEM_COMPLETE

