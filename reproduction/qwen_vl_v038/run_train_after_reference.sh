#!/usr/bin/env bash
set -euo pipefail

PY=/home/zhangzongyuan/anaconda3/envs/radargym-rl/bin/python
ACCELERATE=/home/zhangzongyuan/anaconda3/envs/radargym-rl/bin/accelerate
ROOT=/home/zhangzongyuan/Myproject/drivelm/DriveLM-main
TRAINER=$ROOT/reproduction/qwen_vl_v038/train_grounding_dpo_ddp.py
BASE=/mnt/data/zzy/drivelm/models/Qwen2.5-VL-7B-Instruct
ADAPTER=/mnt/data/zzy/drivelm/models/qwen2.5-vl-7b-drivelm-v037b-anchored-dpo-seed42/checkpoint-75
REFERENCE=/mnt/data/zzy/drivelm/reproduction/qwen_vl_v038/reference/preferences_with_v037b_reference.jsonl
SMOKE=/mnt/data/zzy/drivelm/reproduction/qwen_vl_v038/smoke/train
MODEL=/mnt/data/zzy/drivelm/models/qwen2.5-vl-7b-drivelm-v038a-grounding-dpo-seed42
LOG=/mnt/data/zzy/drivelm/reproduction/qwen_vl_v038/logs
GPUS=GPU-9e49353e-f137-3d6d-f7d5-987593a56d30,GPU-07f03596-83e4-920f-9669-fbec78272f67,GPU-8f2fb421-991d-f5eb-c811-448f7e105297

while tmux has-session -t v038_reference 2>/dev/null; do sleep 15; done
test -s "$REFERENCE"
test -s "$REFERENCE.report.json"

CUDA_VISIBLE_DEVICES=$GPUS "$ACCELERATE" launch \
  --multi_gpu --num_processes 3 --mixed_precision bf16 \
  --main_process_port 29638 "$TRAINER" \
  --experiment-name v038a-grounding-smoke \
  --model-path "$BASE" --adapter-path "$ADAPTER" \
  --reference-jsonl "$REFERENCE" --output-dir "$SMOKE" \
  --max-train-samples 12 --gradient-accumulation-steps 4 \
  --learning-rate 1e-6 --beta 0.05 --chosen-ce-weight 0.1 \
  --max-steps 1 --save-steps 0 --max-pixels 100352 \
  --max-length 4096 --seed 42 > "$LOG/train_smoke.log" 2>&1

"$PY" -c "import json,math,pathlib; lines=pathlib.Path('$LOG/train_smoke.log').read_text().splitlines(); rows=[json.loads(x) for x in lines if x.startswith('{') and '\"step\"' in x]; assert len(rows)==1, len(rows); r=rows[0]; assert abs(r['dpo_logit']) < 0.01, r; assert abs(r['dpo_loss']-math.log(2)) < 0.01, r; assert math.isfinite(r['grad_norm']), r; print(json.dumps({'smoke_passed':True,'step':r},indent=2))" \
  > "$LOG/train_smoke_validation.json"

CUDA_VISIBLE_DEVICES=$GPUS "$ACCELERATE" launch \
  --multi_gpu --num_processes 3 --mixed_precision bf16 \
  --main_process_port 29639 "$TRAINER" \
  --experiment-name v038a-grounding-anchored-dpo-seed42 \
  --model-path "$BASE" --adapter-path "$ADAPTER" \
  --reference-jsonl "$REFERENCE" --output-dir "$MODEL" \
  --gradient-accumulation-steps 4 --learning-rate 1e-6 \
  --beta 0.05 --chosen-ce-weight 0.1 --max-steps 100 \
  --save-steps 25 --max-pixels 100352 --max-length 4096 --seed 42 \
  > "$LOG/train.log" 2>&1

echo V038A_TRAIN_COMPLETE
