#!/usr/bin/env bash
set -euo pipefail

PY=/home/zhangzongyuan/anaconda3/envs/radargym-rl/bin/python
ROOT=/home/zhangzongyuan/Myproject/drivelm/DriveLM-main
HERE=$ROOT/reproduction/qwen_vl_v040_trajectory_rl
RUN=/mnt/data/zzy/drivelm/reproduction/qwen_vl_v040_trajectory_rl
OUT=$RUN/final_eval
CACHE=/mnt/data/zzy/drivelm/reproduction/drivelm_ds/deepseek_judge.sqlite

while [[ ! -s "$OUT/OFFLINE_COMPLETE" ]]; do sleep 15; done
cd "$ROOT/reproduction/drivelm_ds_eval"
for name in baseline grpo70 gspo90; do
  target="$OUT/metrics/${name}_drivelm_ds.json"
  [[ -s "$target" ]] && continue
  attempt=1
  until env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
    "$PY" evaluate.py --references-jsonl "$OUT/planning_references.jsonl" \
      --predictions-json "$OUT/predictions/${name}.json" --output-json "$target" \
      --cache-file "$CACHE" --workers 8 --disable-graph-gating; do
    if (( attempt >= 10 )); then
      echo "${name} semantic evaluation failed after ${attempt} attempts" >&2
      exit 12
    fi
    attempt=$((attempt + 1))
    sleep 30
  done
done
cd "$ROOT"
"$PY" "$HERE/compare_final_eval.py" --eval-dir "$OUT" \
  --output-json "$OUT/final_comparison.json" \
  --output-markdown "$OUT/final_comparison.md"
printf 'V040_SEMANTIC_EVAL_COMPLETE\n' > "$OUT/SEMANTIC_COMPLETE"
