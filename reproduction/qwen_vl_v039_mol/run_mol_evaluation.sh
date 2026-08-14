#!/usr/bin/env bash
set -euo pipefail
PY=/home/zhangzongyuan/anaconda3/envs/radargym-rl/bin/python
ROOT=/home/zhangzongyuan/Myproject/drivelm/DriveLM-main
DEV=/mnt/data/zzy/drivelm/reproduction/qwen_vl_stage2_ablation_v1/qwen_v033_c00_dev.jsonl
DIR=/mnt/data/zzy/drivelm/reproduction/qwen_vl_v039_mol
MOL=$DIR/mol_predictions/mol_dev_predictions.json
SHARED=$DIR/shared_predictions.json
BASE_OFF=/mnt/data/zzy/drivelm/reproduction/qwen_vl_v037b/checkpoint_sweep/checkpoint-75-dev-metrics.json
BASE_ST=/mnt/data/zzy/drivelm/reproduction/qwen_vl_v038/v037b_baseline_structural.json
BASE_DS=/mnt/data/zzy/drivelm/reproduction/qwen_vl_v037b/checkpoint-75-drivelm-ds.json
CACHE=/mnt/data/zzy/drivelm/reproduction/drivelm_ds/deepseek_judge.sqlite
OUT=$DIR/evaluation
mkdir -p "$OUT"
while [[ ! -s "$MOL" || ! -s "$SHARED" ]]; do sleep 15; done
for name in mol shared; do
 pred=$MOL; [[ $name == shared ]] && pred=$SHARED
 "$PY" "$ROOT/reproduction/qwen_vl/evaluate_offline.py" --references-jsonl "$DEV" --predictions-json "$pred" --output-json "$OUT/${name}_offline.json"
 "$PY" "$ROOT/reproduction/qwen_vl_v038/evaluate_structural.py" --references-jsonl "$DEV" --predictions-json "$pred" --output-json "$OUT/${name}_structural.json"
done
"$PY" "$ROOT/reproduction/qwen_vl_v039_mol/select_mol_candidate.py" --baseline-offline "$BASE_OFF" --baseline-structural "$BASE_ST" --shared-offline "$OUT/shared_offline.json" --shared-structural "$OUT/shared_structural.json" --mol-offline "$OUT/mol_offline.json" --mol-structural "$OUT/mol_structural.json" --output-json "$OUT/offline_selection.json"
mapfile -t candidates < <("$PY" -c "import json;print(*json.load(open('$OUT/offline_selection.json'))['judge_candidates'],sep='\\n')")
for name in "${candidates[@]}"; do
 [[ -n $name ]] || continue
 pred=$MOL; [[ $name == shared ]] && pred=$SHARED
 cd "$ROOT/reproduction/drivelm_ds_eval"
 env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy "$PY" evaluate.py --references-jsonl "$DEV" --predictions-json "$pred" --output-json "$OUT/${name}_drivelm_ds.json" --cache-file "$CACHE" --workers 8
 done
cd "$ROOT"
args=()
[[ -s "$OUT/shared_drivelm_ds.json" ]] && args+=(--shared-ds "$OUT/shared_drivelm_ds.json")
[[ -s "$OUT/mol_drivelm_ds.json" ]] && args+=(--mol-ds "$OUT/mol_drivelm_ds.json")
"$PY" "$ROOT/reproduction/qwen_vl_v039_mol/summarize_mol.py" --baseline-ds "$BASE_DS" --selection "$OUT/offline_selection.json" "${args[@]}" --output-json "$OUT/full_summary.json"
echo V039A_MOL_EVALUATION_COMPLETE
