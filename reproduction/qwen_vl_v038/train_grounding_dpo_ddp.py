#!/usr/bin/env python3
"""Grounding-aware balanced DPO initialized from frozen v0.37B checkpoint-75."""

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
import torch.nn.functional as F
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset


HERE = Path(__file__).resolve().parent
COMMON_DIR = HERE.parent / "qwen_vl"
sys.path.insert(0, str(COMMON_DIR))
from common import load_model, load_processor  # noqa: E402
from preference_common import (  # noqa: E402
    close_images,
    encode_answer,
    load_row_images,
    read_preference_jsonl,
    sequence_score,
)


class PreferenceDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def single_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 1:
        raise ValueError("v0.38A DPO requires per-device batch size 1")
    return rows[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-name", default="v038a-grounding-anchored-dpo")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--reference-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume-from")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--chosen-ce-weight", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--min-pixels", type=int, default=32 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=128 * 28 * 28)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-steps", type=int, default=25)
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
    completed_batches: int,
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
                "completed_batches": completed_batches,
                "optimizer": optimizer.state_dict(),
                "metadata": metadata,
            },
            path / "v038a_training_state.pt",
        )
    accelerator.wait_for_everyone()


def main() -> None:
    args = parse_args()
    if args.beta <= 0 or args.gradient_accumulation_steps <= 0:
        raise ValueError("beta and gradient accumulation must be positive")
    if args.chosen_ce_weight < 0:
        raise ValueError("chosen CE weight must be non-negative")
    rows = read_preference_jsonl(args.reference_jsonl, args.max_train_samples)
    normalization = {str(row.get("reference_logp_normalization")) for row in rows}
    if normalization - {"sum", "mean"} or len(normalization) != 1:
        raise ValueError(f"Invalid reference normalization: {normalization}")
    logp_normalization = normalization.pop()
    required = ("reference_chosen_logp", "reference_rejected_logp")
    if any(any(key not in row for key in required) for row in rows):
        raise ValueError("Reference JSONL is missing frozen-policy log probabilities")
    reference_policies = {str(row.get("reference_policy")) for row in rows}
    if len(reference_policies) != 1:
        raise ValueError(f"Mixed reference policies: {reference_policies}")
    reference_policy = reference_policies.pop()

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
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        PreferenceDataset(rows),
        batch_size=1,
        shuffle=False,
        generator=generator,
        num_workers=0,
        collate_fn=single_row,
    )
    processor = load_processor(args.adapter_path, args.min_pixels, args.max_pixels)
    adapter_source = args.resume_from or args.adapter_path
    model = load_model(args.model_path, args.dtype)
    model = PeftModel.from_pretrained(model, adapter_source, is_trainable=True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    resume_state: dict[str, Any] | None = None
    if args.resume_from:
        state_path = Path(args.resume_from) / "v038a_training_state.pt"
        resume_state = torch.load(state_path, map_location="cpu", weights_only=False)
        optimizer.load_state_dict(resume_state["optimizer"])
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    model.train()

    available_updates = math.ceil(
        len(loader) * args.epochs / args.gradient_accumulation_steps
    )
    total_steps = args.max_steps or available_updates
    if total_steps <= 0 or total_steps > available_updates:
        raise ValueError(
            f"max-steps {total_steps} invalid; available updates={available_updates}"
        )
    metadata = {
        "objective": "grounding-balanced DPO plus chosen-answer CE anchor",
        "gradient_method": "exact sequential DPO gradient with detached analytic coefficients",
        "policy_initialization": str(Path(adapter_source).resolve()),
        "reference_policy": reference_policy,
        "reference_jsonl": str(Path(args.reference_jsonl).resolve()),
        "reference_sha256": sha256(args.reference_jsonl),
        "reference_logp_normalization": logp_normalization,
        "records": len(rows),
        "world_size": accelerator.num_processes,
        "effective_global_batch_size": (
            accelerator.num_processes * args.gradient_accumulation_steps
        ),
        "beta": args.beta,
        "chosen_ce_weight": args.chosen_ce_weight,
        "learning_rate": args.learning_rate,
        "max_pixels": args.max_pixels,
        "max_length": args.max_length,
        "seed": args.seed,
        "dropout_disabled": True,
    }
    if accelerator.is_main_process:
        print(json.dumps({"training_configuration": metadata}, indent=2), flush=True)

    global_step = int(resume_state["global_step"]) if resume_state else 0
    completed_batches = int(resume_state["completed_batches"]) if resume_state else 0
    started = time.time()
    cumulative_loss = 0.0
    cumulative_accuracy = 0.0
    stop = False
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        for batch_index, row in enumerate(loader):
            absolute_batch = epoch * len(loader) + batch_index
            if absolute_batch < completed_batches:
                continue
            images = load_row_images(row)
            try:
                chosen_batch, chosen_tokens = encode_answer(
                    row, row["chosen"], processor, images,
                    args.max_length, accelerator.device,
                )
                rejected_batch, rejected_tokens = encode_answer(
                    row, row["rejected"], processor, images,
                    args.max_length, accelerator.device,
                )
                with accelerator.accumulate(model):
                    with torch.no_grad():
                        current_chosen = sequence_score(
                            model, chosen_batch, chosen_tokens, logp_normalization
                        )
                        current_rejected = sequence_score(
                            model, rejected_batch, rejected_tokens, logp_normalization
                        )
                        reference_margin = torch.tensor(
                            float(row["reference_chosen_logp"])
                            - float(row["reference_rejected_logp"]),
                            device=accelerator.device,
                        )
                        policy_margin = current_chosen - current_rejected
                        logit = args.beta * (policy_margin - reference_margin)
                        dpo_loss = -F.logsigmoid(logit)
                        chosen_ce = (
                            -current_chosen / chosen_tokens
                            if logp_normalization == "sum"
                            else -current_chosen
                        )
                        loss = dpo_loss + args.chosen_ce_weight * chosen_ce
                        sigmoid_negative = torch.sigmoid(-logit).detach()
                        ce_score_coefficient = (
                            -args.chosen_ce_weight / chosen_tokens
                            if logp_normalization == "sum"
                            else -args.chosen_ce_weight
                        )
                        chosen_coefficient = (
                            -args.beta * sigmoid_negative + ce_score_coefficient
                        )
                        rejected_coefficient = args.beta * sigmoid_negative

                    chosen_score = sequence_score(
                        model, chosen_batch, chosen_tokens, logp_normalization
                    )
                    accelerator.backward(chosen_score * chosen_coefficient)
                    del chosen_score
                    rejected_score = sequence_score(
                        model, rejected_batch, rejected_tokens, logp_normalization
                    )
                    accelerator.backward(rejected_score * rejected_coefficient)
                    del rejected_score
                    if accelerator.sync_gradients:
                        grad_norm = accelerator.clip_grad_norm_(
                            model.parameters(), args.max_grad_norm
                        )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            finally:
                close_images(images)
            completed_batches = absolute_batch + 1
            if not accelerator.sync_gradients:
                continue
            global_step += 1
            mean_loss = accelerator.gather(loss.detach().reshape(1)).mean().item()
            preference_accuracy = accelerator.gather(
                (policy_margin > reference_margin).float().reshape(1)
            ).mean().item()
            cumulative_loss += mean_loss
            cumulative_accuracy += preference_accuracy
            if accelerator.is_main_process:
                print(
                    json.dumps(
                        {
                            "step": global_step,
                            "total_steps": total_steps,
                            "loss": mean_loss,
                            "dpo_loss": float(dpo_loss.detach().cpu()),
                            "chosen_ce": float(chosen_ce.detach().cpu()),
                            "mean_loss": cumulative_loss / global_step,
                            "preference_accuracy": preference_accuracy,
                            "mean_preference_accuracy": cumulative_accuracy / global_step,
                            "policy_margin": float(policy_margin.detach().cpu()),
                            "reference_margin": float(reference_margin.detach().cpu()),
                            "dpo_logit": float(logit.detach().cpu()),
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
                    completed_batches,
                    metadata,
                )
            if global_step >= total_steps:
                stop = True
                break
        if stop:
            break

    output_dir = Path(args.output_dir)
    save_checkpoint(
        accelerator,
        output_dir,
        model,
        processor,
        optimizer,
        global_step,
        completed_batches,
        metadata,
    )
    if accelerator.is_main_process:
        report = {
            "experiment": args.experiment_name,
            "output_dir": str(output_dir.resolve()),
            "global_steps": global_step,
            "completed_batches_per_process": completed_batches,
            "mean_loss": cumulative_loss / max(global_step, 1),
            "mean_preference_accuracy": cumulative_accuracy / max(global_step, 1),
            "elapsed_sec": round(time.time() - started, 2),
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
