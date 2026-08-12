#!/usr/bin/env python3
"""Pure assistant-token CE LoRA SFT for single-frame six-camera DriveLM."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader


HERE = Path(__file__).resolve().parent
CAMERA_ONLY_DIR = HERE.parent / "qwen_vl"
sys.path.insert(0, str(CAMERA_ONLY_DIR))
from common import DriveLMDataset, SixCameraCollator, load_model, load_processor, read_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-name", default="v036-camera-only-ce")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--min-pixels", type=int, default=32 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=128 * 28 * 28)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(
    accelerator: Accelerator,
    path: Path,
    model: torch.nn.Module,
    processor: Any,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    args: argparse.Namespace,
    metadata: dict[str, Any],
) -> None:
    accelerator.wait_for_everyone()
    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        path.mkdir(parents=True, exist_ok=True)
        accelerator.unwrap_model(model).save_pretrained(
            path,
            state_dict=state_dict,
            save_function=accelerator.save,
            safe_serialization=True,
        )
        processor.save_pretrained(path)
        torch.save(
            {
                "global_step": global_step,
                "optimizer": optimizer.state_dict(),
                "seed": args.seed,
                "world_size": accelerator.num_processes,
                "per_device_batch_size": args.per_device_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "metadata": metadata,
            },
            path / "v036_training_state.pt",
        )
    accelerator.wait_for_everyone()


def main() -> None:
    args = parse_args()
    if args.per_device_batch_size <= 0 or args.gradient_accumulation_steps <= 0:
        raise ValueError("batch size and gradient accumulation must be positive")
    if args.max_train_samples < 0 or args.max_steps < 0:
        raise ValueError("sample and step limits must be non-negative")

    records = read_jsonl(args.train_jsonl, args.max_train_samples)
    processor = load_processor(args.model_path, args.min_pixels, args.max_pixels)
    collator = SixCameraCollator(processor, max_length=args.max_length)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        DriveLMDataset(records),
        batch_size=args.per_device_batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        collate_fn=collator,
    )

    if args.dry_run:
        batch = next(iter(loader))
        grid = batch["image_grid_thw"]
        report = {
            "objective": "pure assistant-token autoregressive cross entropy",
            "records": len(records),
            "input_ids_shape": list(batch["input_ids"].shape),
            "pixel_values_shape": list(batch["pixel_values"].shape),
            "image_grid_thw_shape": list(grid.shape),
            "visual_inputs": int(grid.shape[0]),
            "approx_merged_visual_tokens": int(
                (grid[:, 0] * grid[:, 1] * grid[:, 2] / 4).sum().item()
            ),
            "supervised_tokens": int((batch["labels"] != -100).sum()),
            "ignored_tokens": int((batch["labels"] == -100).sum()),
            "max_pixels_per_image": args.max_pixels,
            "max_length": args.max_length,
        }
        if report["visual_inputs"] != 6 * args.per_device_batch_size:
            raise AssertionError(
                f"Expected exactly six images per sample, got {report['visual_inputs']}"
            )
        print(json.dumps(report, indent=2))
        return

    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=False,
        broadcast_buffers=False,
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.dtype,
        kwargs_handlers=[ddp_kwargs],
    )
    set_seed(args.seed)
    model = load_model(args.model_path, args.dtype)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        ),
    )
    if accelerator.is_main_process:
        model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    available_updates = math.floor(
        len(loader) * args.epochs / args.gradient_accumulation_steps
    )
    total_steps = args.max_steps or available_updates
    if total_steps <= 0 or total_steps > available_updates:
        raise ValueError(
            f"max-steps {total_steps} is invalid; available updates={available_updates}"
        )

    metadata = {
        "objective": "pure assistant-token autoregressive cross entropy",
        "loss_weighting": "none: no tag, metric, graph-gating, coordinate, or judge weights",
        "input_contract": "one current frame with exactly six synchronized cameras",
        "camera_order": [
            "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
            "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
        ],
        "train_manifest_sha256": sha256(args.train_jsonl),
        "records": len(records),
        "available_updates": available_updates,
        "total_steps": total_steps,
        "effective_global_batch_size": (
            args.per_device_batch_size
            * accelerator.num_processes
            * args.gradient_accumulation_steps
        ),
        "max_pixels_per_image": args.max_pixels,
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "seed": args.seed,
    }
    if accelerator.is_main_process:
        print(json.dumps({"training_configuration": metadata}, indent=2), flush=True)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    started = time.time()
    global_step = 0
    cumulative_loss = 0.0
    for _epoch in range(args.epochs):
        for batch in loader:
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if not accelerator.sync_gradients:
                continue
            global_step += 1
            mean_loss = accelerator.gather(loss.detach().reshape(1)).mean().item()
            cumulative_loss += mean_loss
            if accelerator.is_main_process:
                print(
                    json.dumps(
                        {
                            "step": global_step,
                            "total_steps": total_steps,
                            "loss": mean_loss,
                            "mean_loss": cumulative_loss / global_step,
                            "grad_norm": float(grad_norm.detach().cpu()),
                            "elapsed_sec": round(time.time() - started, 2),
                        }
                    ),
                    flush=True,
                )
            if args.save_steps > 0 and global_step % args.save_steps == 0:
                save_checkpoint(
                    accelerator,
                    Path(args.output_dir) / f"checkpoint-{global_step}",
                    model,
                    processor,
                    optimizer,
                    global_step,
                    args,
                    metadata,
                )
            if global_step >= total_steps:
                break
        if global_step >= total_steps:
            break

    output_dir = Path(args.output_dir)
    save_checkpoint(
        accelerator,
        output_dir,
        model,
        processor,
        optimizer,
        global_step,
        args,
        metadata,
    )
    elapsed = time.time() - started
    if accelerator.is_main_process:
        report = {
            "experiment": args.experiment_name,
            "base_model": str(Path(args.model_path).resolve()),
            "train_jsonl": str(Path(args.train_jsonl).resolve()),
            "output_dir": str(output_dir.resolve()),
            "global_steps": global_step,
            "mean_loss": cumulative_loss / max(global_step, 1),
            "elapsed_sec": round(elapsed, 2),
            "seconds_per_update": round(elapsed / max(global_step, 1), 4),
            "world_size": accelerator.num_processes,
            "per_device_batch_size": args.per_device_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            **metadata,
            "hostname": os.uname().nodename,
        }
        (output_dir / "training_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2), flush=True)
    accelerator.end_training()


if __name__ == "__main__":
    main()
