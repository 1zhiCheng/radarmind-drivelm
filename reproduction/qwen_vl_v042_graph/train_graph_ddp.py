#!/usr/bin/env python3
"""Three-GPU multi-turn Graph-SFT for DriveLM with pure assistant-token CE."""

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
from peft import PeftModel
from torch.utils.data import DataLoader


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "qwen_vl"))
from common import load_model, load_processor  # noqa: E402
from graph_data import (  # noqa: E402
    GraphTrajectoryCollator,
    GraphTrajectoryDataset,
    read_graph_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-name", default="v042-graph-sft")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--min-pixels", type=int, default=32 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=128 * 28 * 28)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
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
                "metadata": metadata,
            },
            path / "v042_training_state.pt",
        )
    accelerator.wait_for_everyone()


def main() -> None:
    args = parse_args()
    if min(args.per_device_batch_size, args.gradient_accumulation_steps) <= 0:
        raise ValueError("batch size and gradient accumulation must be positive")
    records = read_graph_jsonl(args.train_jsonl, args.max_train_samples)
    adapter_path = args.resume_from_checkpoint or args.adapter_path
    processor = load_processor(adapter_path, args.min_pixels, args.max_pixels)
    collator = GraphTrajectoryCollator(processor, max_length=args.max_length)
    loader = DataLoader(
        GraphTrajectoryDataset(records),
        batch_size=args.per_device_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=args.num_workers,
        collate_fn=collator,
    )

    if args.dry_run:
        batch = next(iter(loader))
        grid = batch["image_grid_thw"]
        report = {
            "objective": "multi-turn pure assistant-token autoregressive cross entropy",
            "graph_contract": "perception->prediction->planning->behavior in one causal sequence",
            "trajectories": len(records),
            "input_ids_shape": list(batch["input_ids"].shape),
            "pixel_values_shape": list(batch["pixel_values"].shape),
            "visual_inputs": int(grid.shape[0]),
            "approx_merged_visual_tokens": int(
                (grid[:, 0] * grid[:, 1] * grid[:, 2] / 4).sum().item()
            ),
            "supervised_tokens": int((batch["labels"] != -100).sum()),
            **collator.last_stats,
        }
        if report["visual_inputs"] != 6 * args.per_device_batch_size:
            raise AssertionError("Graph trajectory must encode exactly six images once")
        print(json.dumps(report, indent=2))
        return

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.dtype,
        kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=False, broadcast_buffers=False)
        ],
    )
    set_seed(args.seed)
    model = load_model(args.model_path, args.dtype)
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    if accelerator.is_main_process:
        model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    start_step = 0
    if args.resume_from_checkpoint:
        state_path = Path(args.resume_from_checkpoint) / "v042_training_state.pt"
        if not state_path.is_file():
            raise FileNotFoundError(f"Missing resumable state: {state_path}")
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        start_step = int(state["global_step"])
        optimizer.load_state_dict(state["optimizer"])

    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    available_updates = math.floor(len(loader) * args.epochs / args.gradient_accumulation_steps)
    total_steps = args.max_steps or available_updates
    if total_steps <= start_step or total_steps > available_updates:
        raise ValueError(
            f"max-steps {total_steps} invalid: start={start_step}, available={available_updates}"
        )
    metadata = {
        "schema_version": "radarmind-drivelm-v042-graph-sft-v1",
        "policy_initialization": str(Path(args.adapter_path).resolve()),
        "resumed_from": str(Path(args.resume_from_checkpoint).resolve()) if args.resume_from_checkpoint else None,
        "objective": "pure CE over every assistant node in a frame-level multi-turn graph",
        "loss_weighting": "token-average CE; no metric, judge, graph-gating, or reward weights",
        "causal_graph": "six cameras once; P answers condition Prediction, then Planning, then Behavior",
        "teacher_forcing": True,
        "train_manifest_sha256": sha256(args.train_jsonl),
        "trajectories": len(records),
        "qa_nodes": sum(len(record["nodes"]) for record in records),
        "available_updates": available_updates,
        "total_steps": total_steps,
        "effective_global_trajectory_batch": (
            args.per_device_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
        ),
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "max_pixels_per_image": args.max_pixels,
        "seed": args.seed,
    }
    if accelerator.is_main_process:
        print(json.dumps({"training_configuration": metadata}, indent=2), flush=True)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    started = time.time()
    global_step = start_step
    loss_sum = 0.0
    for _epoch in range(args.epochs):
        for batch_index, batch in enumerate(loader):
            if batch_index < start_step * args.gradient_accumulation_steps:
                continue
            with accelerator.accumulate(model):
                loss = model(**batch).loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if not accelerator.sync_gradients:
                continue
            global_step += 1
            mean_loss = accelerator.gather(loss.detach().reshape(1)).mean().item()
            loss_sum += mean_loss
            if accelerator.is_main_process:
                print(
                    json.dumps(
                        {
                            "step": global_step,
                            "total_steps": total_steps,
                            "loss": mean_loss,
                            "mean_loss_since_resume": loss_sum / max(global_step - start_step, 1),
                            "grad_norm": float(grad_norm.detach().cpu()),
                            "elapsed_sec": round(time.time() - started, 2),
                        }
                    ),
                    flush=True,
                )
            if args.save_steps and global_step % args.save_steps == 0:
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
    save_checkpoint(accelerator, output_dir, model, processor, optimizer, global_step, args, metadata)
    if accelerator.is_main_process:
        elapsed = time.time() - started
        report = {
            "experiment": args.experiment_name,
            "global_steps": global_step,
            "mean_loss_since_resume": loss_sum / max(global_step - start_step, 1),
            "elapsed_sec": round(elapsed, 2),
            "seconds_per_update": round(elapsed / max(global_step - start_step, 1), 4),
            "world_size": accelerator.num_processes,
            "hostname": os.uname().nodename,
            **metadata,
        }
        (output_dir / "training_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2), flush=True)
    accelerator.end_training()


if __name__ == "__main__":
    main()
