#!/usr/bin/env python3
"""Single-GPU LoRA SFT for the DriveLM six-camera Qwen2.5-VL baseline."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader

from common import DriveLMDataset, SixCameraCollator, load_model, load_processor, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--min-pixels", type=int, default=32 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=96 * 28 * 28)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--resume-adapter")
    parser.add_argument("--initial-step", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    records = read_jsonl(args.train_jsonl, args.max_train_samples)
    processor = load_processor(args.model_path, args.min_pixels, args.max_pixels)
    collator = SixCameraCollator(processor, max_length=args.max_length)
    loader = DataLoader(
        DriveLMDataset(records), batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collator,
    )
    if args.dry_run:
        batch = next(iter(loader))
        report = {
            "records": len(records),
            "input_ids_shape": list(batch["input_ids"].shape),
            "pixel_values_shape": list(batch["pixel_values"].shape),
            "image_grid_thw_shape": list(batch["image_grid_thw"].shape),
            "supervised_tokens": int((batch["labels"] != -100).sum()),
        }
        print(json.dumps(report, indent=2))
        return

    model = load_model(args.model_path, args.dtype)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    if args.resume_adapter:
        model = PeftModel.from_pretrained(
            model, args.resume_adapter, is_trainable=True,
        )
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                r=args.lora_rank, lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout, bias="none", task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ),
        )
    model.print_trainable_parameters()
    model.to(args.device)
    model.train()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    total_steps = args.max_steps or math.ceil(len(loader) * args.epochs / args.gradient_accumulation_steps)
    started = time.time()
    global_step = args.initial_step
    updates_this_run = 0
    micro_step = 0
    loss_sum = 0.0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        for batch in loader:
            batch = {key: value.to(args.device) for key, value in batch.items()}
            raw_loss = model(**batch).loss
            (raw_loss / args.gradient_accumulation_steps).backward()
            micro_step += 1
            if micro_step % args.gradient_accumulation_steps:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            updates_this_run += 1
            loss_sum += float(raw_loss.detach().cpu())
            print(json.dumps({"step": global_step, "total_steps": total_steps, "loss_this_run": loss_sum / updates_this_run}), flush=True)
            if args.save_steps > 0 and global_step % args.save_steps == 0:
                checkpoint_dir = Path(args.output_dir) / f"checkpoint-{global_step}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(checkpoint_dir)
            if global_step >= total_steps:
                break
        if global_step >= total_steps:
            break
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    report = {
        "base_model": args.model_path, "train_jsonl": args.train_jsonl,
        "records_loaded": len(records), "global_steps": global_step,
        "initial_step": args.initial_step,
        "updates_this_run": updates_this_run,
        "mean_update_loss_this_run": loss_sum / max(updates_this_run, 1),
        "resume_adapter": args.resume_adapter,
        "elapsed_sec": round(time.time() - started, 2),
        "six_camera": True, "max_pixels_per_image": args.max_pixels,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
    }
    (output_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
