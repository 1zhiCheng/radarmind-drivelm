# v0.36 camera-only pure-CE model and resolution study

This experiment is isolated from C00-CE, C00-OA, and all historical/radar
branches. Every sample contains one current timestamp and exactly six
synchronized camera images. The trainer imports the original camera-only
collator, so no history/ROI prompt remains.

The SFT objective is the model's standard autoregressive cross entropy over
assistant tokens. Image, system, user, padding, task-tag, metric, graph-gating,
coordinate, and semantic-judge weights are absent.

## Dry run

```bash
cd /home/zhangzongyuan/Myproject/drivelm/DriveLM-main
PY=/home/zhangzongyuan/anaconda3/envs/radargym-rl/bin/python
MODEL=/public/zzy/RadarMind/models/Qwen2.5-VL-3B-Instruct
TRAIN=/mnt/data/zzy/drivelm/reproduction/qwen_vl_stage2_ablation_v1/qwen_v033_c00_train.jsonl

$PY reproduction/qwen_vl_v036/train_ddp.py \
  --model-path "$MODEL" --train-jsonl "$TRAIN" --output-dir /tmp/v036-dry-run \
  --max-train-samples 64 --per-device-batch-size 1 \
  --max-pixels $((128*28*28)) --max-length 4096 --dry-run
```

The report must show exactly six visual inputs for one sample and at least one
supervised assistant token.

## Two-RTX-5090 controlled launch

```bash
CUDA_VISIBLE_DEVICES=0,2 accelerate launch --multi_gpu --num_processes 2 \
  reproduction/qwen_vl_v036/train_ddp.py \
  --experiment-name v036-b00-qwen25vl3b-128 \
  --model-path "$MODEL" --train-jsonl "$TRAIN" \
  --output-dir /mnt/data/zzy/drivelm/models/qwen2.5-vl-3b-drivelm-v036-b00-seed42 \
  --per-device-batch-size 1 --gradient-accumulation-steps 2 \
  --epochs 1 --learning-rate 2e-4 --lora-rank 8 --lora-alpha 16 \
  --max-pixels $((128*28*28)) --max-length 4096 --save-steps 500
```

For B10, only the base model changes. The controlled capacity comparison keeps
the prompt, data, visual budget, LoRA, batch size, and optimization
hyperparameters equal. B10 is the required end-to-end pipeline target. B11
(256 visual tokens/image) is optional and runs only if B10 diagnostics justify
a higher visual budget. B12 is cancelled.
