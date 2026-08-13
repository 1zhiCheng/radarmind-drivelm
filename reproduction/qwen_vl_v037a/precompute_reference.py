#!/usr/bin/env python3
"""Precompute frozen-B10 chosen and rejected log probabilities."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--preference-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--skip-reference-jsonl", action="append", default=[])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-shard-records", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--min-pixels", type=int, default=32 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=128 * 28 * 28)
    parser.add_argument("--normalization", choices=("sum", "mean"), default="sum")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_completed(path: Path) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return completed
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = str(row["id"])
            if row_id in completed:
                raise ValueError(f"Duplicate id {row_id} in {path}:{line_number}")
            completed[row_id] = row
    return completed


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard configuration")
    rows = read_preference_jsonl(args.preference_jsonl)
    skip_ids: set[str] = set()
    for skip_path in args.skip_reference_jsonl:
        with Path(skip_path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    skip_ids.add(str(json.loads(line)["id"]))
    shard_rows = [
        row for index, row in enumerate(rows)
        if index % args.num_shards == args.shard_index
        and str(row["id"]) not in skip_ids
    ]
    if args.max_shard_records:
        shard_rows = shard_rows[: args.max_shard_records]
    expected_ids = {str(row["id"]) for row in shard_rows}
    output = Path(args.output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = read_completed(output) if args.resume else {}
    if set(completed) - expected_ids:
        raise ValueError("Resume output contains ids outside this reference shard")

    processor = load_processor(args.adapter_path, args.min_pixels, args.max_pixels)
    model = load_model(args.model_path, args.dtype)
    model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=False)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    model.to(args.device).train()
    autocast_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    started = time.time()
    mode = "a" if args.resume else "w"
    with output.open(mode, encoding="utf-8") as handle:
        for row in shard_rows:
            row_id = str(row["id"])
            if row_id in completed:
                continue
            images = load_row_images(row)
            try:
                chosen_batch, chosen_tokens = encode_answer(
                    row, row["chosen"], processor, images, args.max_length, args.device
                )
                rejected_batch, rejected_tokens = encode_answer(
                    row, row["rejected"], processor, images, args.max_length, args.device
                )
                with torch.inference_mode(), torch.autocast(
                    device_type="cuda", dtype=autocast_dtype
                ):
                    chosen_score = sequence_score(
                        model, chosen_batch, chosen_tokens, args.normalization
                    )
                    rejected_score = sequence_score(
                        model, rejected_batch, rejected_tokens, args.normalization
                    )
            finally:
                close_images(images)
            enriched = {
                **row,
                "reference_policy": "B10",
                "reference_logp_normalization": args.normalization,
                "reference_chosen_logp": float(chosen_score.item()),
                "reference_rejected_logp": float(rejected_score.item()),
                "reference_chosen_tokens": chosen_tokens,
                "reference_rejected_tokens": rejected_tokens,
            }
            handle.write(json.dumps(enriched, ensure_ascii=False) + "\n")
            handle.flush()
            completed[row_id] = enriched
            print(
                json.dumps(
                    {
                        "shard": args.shard_index,
                        "done": len(completed),
                        "total": len(shard_rows),
                        "id": row_id,
                        "reference_margin": enriched["reference_chosen_logp"]
                        - enriched["reference_rejected_logp"],
                    }
                ),
                flush=True,
            )

    adapter_weights = Path(args.adapter_path) / "adapter_model.safetensors"
    report = {
        "schema_version": "drivelm-v037a-reference-logp-v1",
        "reference_policy": "B10 frozen",
        "preference_jsonl": str(Path(args.preference_jsonl).resolve()),
        "preference_sha256": sha256(args.preference_jsonl),
        "adapter_sha256": sha256(adapter_weights),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "records_expected": len(shard_rows),
        "records_complete": len(completed),
        "precomputed_ids_skipped": len(skip_ids),
        "normalization": args.normalization,
        "execution_mode": "train mode with dropout disabled to match DPO policy",
        "external_api_used": False,
        "elapsed_seconds": time.time() - started,
    }
    report_path = output.with_suffix(output.suffix + ".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "report": str(report_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
