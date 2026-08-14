#!/usr/bin/env bash
set -euo pipefail

PY=/home/zhangzongyuan/anaconda3/envs/radargym-rl/bin/python
ROOT=/home/zhangzongyuan/Myproject/drivelm/DriveLM-main
HERE=$ROOT/reproduction/qwen_vl_v039_mol
DIR=/mnt/data/zzy/drivelm/reproduction/qwen_vl_v039_mol/sweep
EVAL=$DIR/evaluation
DEV=/mnt/data/zzy/drivelm/reproduction/qwen_vl_stage2_ablation_v1/qwen_v033_c00_dev.jsonl
BASE_PRED=/mnt/data/zzy/drivelm/reproduction/qwen_vl_v039_mol/mol_predictions/mol_dev_predictions.json
BASE_DS=/mnt/data/zzy/drivelm/reproduction/qwen_vl_v039_mol/evaluation/mol_drivelm_ds.json
BASE_OFF=/mnt/data/zzy/drivelm/reproduction/qwen_vl_v039_mol/evaluation/mol_offline.json
BASE_ST=/mnt/data/zzy/drivelm/reproduction/qwen_vl_v039_mol/evaluation/mol_structural.json
CACHE=/mnt/data/zzy/drivelm/reproduction/drivelm_ds/deepseek_judge.sqlite
MODEL_ROOT=/mnt/data/zzy/drivelm/models/qwen2.5-vl-7b-drivelm-v039b-mol-sweep
V039A_ROOT=/mnt/data/zzy/drivelm/models/qwen2.5-vl-7b-drivelm-v039a-mol-pilot
mkdir -p "$DIR"

# The first 100..500 probe may already be running. Files are the source of
# truth, so controller restarts are idempotent.
while [[ ! -s "$DIR/monitor_report.json" ]]; do sleep 15; done

while true; do
  decision=$($PY -c "import json; print(json.load(open('$DIR/monitor_report.json'))['decision'])")
  [[ $decision == extend_sweep ]] || break
  latest=$($PY - <<PY
from pathlib import Path
p=Path('$EVAL')
print(max(int(x.name.split('-')[1].split('_')[0]) for x in p.glob('checkpoint-*_offline.json')))
PY
)
  (( latest < 1500 )) || break
  target=$((latest + 200)); (( target > 1500 )) && target=1500
  "$HERE/continue_expert_sweep.sh" "$latest" "$target"
  points=(); for ((s=latest+100; s<=target; s+=100)); do points+=("$s"); done
  "$HERE/evaluate_sweep_points.sh" "${points[@]}"
  mapfile -t all_steps < <($PY - <<PY
from pathlib import Path
for s in sorted(int(x.name.split('-')[1].split('_')[0]) for x in Path('$EVAL').glob('checkpoint-*_offline.json')): print(s)
PY
)
  "$PY" "$HERE/monitor_sweep.py" --evaluation-dir "$EVAL" \
    --baseline-offline "$BASE_OFF" --baseline-structural "$BASE_ST" \
    --steps "${all_steps[@]}" --min-delta 0.0005 --patience 2 --max-step-cap 1500 \
    --output-json "$DIR/monitor_report.json"
done

best_step=$($PY -c "import json; print(json.load(open('$DIR/monitor_report.json'))['best_step'] or '')")
if [[ -z $best_step ]]; then
  "$PY" "$HERE/finalize_adaptive_sweep.py" --monitor "$DIR/monitor_report.json" \
    --full-baseline "$BASE_DS" --model-root "$MODEL_ROOT" --v039a-root "$V039A_ROOT" \
    --output-json "$DIR/best_checkpoint.json"
  echo V039B_ADAPTIVE_CONTROLLER_COMPLETE
  exit 0
fi

PRED="$DIR/checkpoint-${best_step}/mol_dev_predictions.json"
FULL_DS="$EVAL/checkpoint-${best_step}_drivelm_ds.json"
cd "$ROOT/reproduction/drivelm_ds_eval"
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  "$PY" evaluate.py --references-jsonl "$DEV" --predictions-json "$PRED" \
  --output-json "$FULL_DS" --cache-file "$CACHE" --workers 8
cd "$ROOT"

PAIR="$DIR/final_common_gating"
"$PY" "$ROOT/reproduction/qwen_vl_v037b/build_common_gating_subset.py" \
  --references-jsonl "$DEV" --baseline-predictions "$BASE_PRED" \
  --candidate-predictions "$PRED" --output-dir "$PAIR"
for name in baseline candidate; do
  "$PY" "$ROOT/reproduction/qwen_vl/evaluate_offline.py" \
    --references-jsonl "$PAIR/common_references.jsonl" \
    --predictions-json "$PAIR/${name}_predictions.json" --output-json "$PAIR/${name}_offline.json"
  cd "$ROOT/reproduction/drivelm_ds_eval"
  env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
    "$PY" evaluate.py --references-jsonl "$PAIR/common_references.jsonl" \
    --predictions-json "$PAIR/${name}_predictions.json" \
    --output-json "$PAIR/${name}_drivelm_ds.json" --cache-file "$CACHE" --workers 8
  cd "$ROOT"
done
"$PY" "$ROOT/reproduction/qwen_vl_v037b/summarize_common_gating.py" \
  --subset-report "$PAIR/common_subset_report.json" \
  --baseline-offline "$PAIR/baseline_offline.json" --candidate-offline "$PAIR/candidate_offline.json" \
  --baseline-drivelm-ds "$PAIR/baseline_drivelm_ds.json" --candidate-drivelm-ds "$PAIR/candidate_drivelm_ds.json" \
  --output-json "$PAIR/common_gating_comparison.json"
"$PY" "$HERE/finalize_adaptive_sweep.py" --monitor "$DIR/monitor_report.json" \
  --full-baseline "$BASE_DS" --full-candidate "$FULL_DS" \
  --paired-comparison "$PAIR/common_gating_comparison.json" \
  --model-root "$MODEL_ROOT" --v039a-root "$V039A_ROOT" \
  --output-json "$DIR/best_checkpoint.json"
echo V039B_ADAPTIVE_CONTROLLER_COMPLETE
