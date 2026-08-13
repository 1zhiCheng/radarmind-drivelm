# v0.37B conservative DPO reproduction

This stage starts from the frozen v0.36 B10 adapter and changes only preference sampling and the post-training objective. It uses 4,104 balanced train-only pairs, DPO with `beta=0.05`, and a chosen-answer CE anchor with weight `0.1`.

## Prerequisites

- Base model: Qwen2.5-VL-7B-Instruct
- Policy adapter: v0.36 B10
- v0.37A audited preference JSONL with frozen-reference log-probabilities
- Scene-isolated 3,355-QA dev JSONL
- Three same-generation GPUs for DDP training; the recorded run used 3 × RTX 5090

No Hugging Face or DeepSeek token is stored in this directory.

## 1. Build balanced preferences

```bash
python reproduction/qwen_vl_v037b/build_balanced_preferences.py \
  --input-jsonl data/v037a/preferences_with_reference.jsonl \
  --output-jsonl data/v037b/balanced_preferences.jsonl \
  --per-task 1026
```

Audit the generated report before training. Expected total: 4,104 records, exactly 1,026 for each of perception, prediction, planning and behavior, with no duplicate IDs.

## 2. Train on three GPUs

```bash
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch \
  --multi_gpu --num_processes 3 --mixed_precision bf16 \
  reproduction/qwen_vl_v037b/train_anchored_dpo_ddp.py \
  --experiment-name v037b-b10-balanced-anchored-dpo-seed42 \
  --model-path models/Qwen2.5-VL-7B-Instruct \
  --adapter-path models/qwen2.5-vl-7b-drivelm-v036-b10-seed42 \
  --reference-jsonl data/v037b/balanced_preferences.jsonl \
  --output-dir models/qwen2.5-vl-7b-drivelm-v037b \
  --gradient-accumulation-steps 4 --learning-rate 1e-6 \
  --beta 0.05 --chosen-ce-weight 0.1 \
  --max-steps 100 --save-steps 25 \
  --max-pixels 100352 --max-length 4096 --seed 42
```

The implementation keeps the round-robin data order. With world size 3 and gradient accumulation 4, every optimizer update sees 12 samples: 3 from each task.

## 3. Full checkpoint sweep

Use `run_checkpoint_eval.sh` as the recorded multi-GPU orchestration template, or call the common inference/evaluation tools for each checkpoint:

```bash
python reproduction/qwen_vl/infer.py \
  --model-path models/Qwen2.5-VL-7B-Instruct \
  --adapter-path models/qwen2.5-vl-7b-drivelm-v037b/checkpoint-75 \
  --input-jsonl data/qwen_dev.jsonl \
  --output-json outputs/checkpoint-75-dev-predictions.json \
  --device cuda:0 --batch-size 16 --max-new-tokens 256 \
  --max-pixels 100352 --resume

python reproduction/qwen_vl/evaluate_offline.py \
  --references-jsonl data/qwen_dev.jsonl \
  --predictions-json outputs/checkpoint-75-dev-predictions.json \
  --output-json outputs/checkpoint-75-dev-metrics.json
```

Run this for steps 25, 50, 75 and 100. Do not select a checkpoint from partial coverage.

## 4. Frozen selection and semantic evaluation

```bash
python reproduction/qwen_vl_v037b/select_offline_candidate.py \
  --baseline outputs/b10_dev_metrics.json \
  --sweep-dir outputs/v037b_sweep \
  --output-json outputs/v037b_offline_selection.json
```

Only a candidate with 100% coverage, Token-F1 and planning Token-F1 not below B10, and MC strictly above B10 is eligible for the semantic judge. `run_post_eval.sh` documents the recorded DeepSeek retry/cache flow and applies the final frozen promotion gates. Replace its machine-specific paths before reuse.

Expected selected checkpoint for the recorded seed-42 run: step 75. See the [full report](../../docs/current/VERSION_0_37B_DRIVELM_CONSERVATIVE_DPO.md).
