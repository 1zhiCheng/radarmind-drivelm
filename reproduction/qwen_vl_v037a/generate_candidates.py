#!/usr/bin/env python3
"""Generate sampled B10 candidates from DriveLM train records only."""

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
from common import load_images, load_model, load_processor, message_text, qwen_messages, read_jsonl  # noqa: E402
from infer import format_challenge_answer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-shard-records", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--candidates-per-record", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--min-pixels", type=int, default=32 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=128 * 28 * 28)
    parser.add_argument("--seed", type=int, default=42)
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
            item = json.loads(line)
            item_id = str(item["id"])
            if item_id in completed:
                raise ValueError(f"Duplicate id {item_id} in {path}:{line_number}")
            completed[item_id] = item
    return completed


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, num-shards)")
    if args.candidates_per_record <= 0:
        raise ValueError("--candidates-per-record must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive for sampled generation")

    all_records = read_jsonl(args.train_jsonl)
    indexed_records = [
        (index, record)
        for index, record in enumerate(all_records)
        if index % args.num_shards == args.shard_index
    ]
    if args.max_shard_records:
        indexed_records = indexed_records[: args.max_shard_records]
    expected_ids = {str(record["id"]) for _, record in indexed_records}

    output = Path(args.output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = read_completed(output) if args.resume else {}
    unexpected = set(completed) - expected_ids
    if unexpected:
        raise ValueError(f"Resume file contains {len(unexpected)} ids outside this shard")
    mode = "a" if args.resume else "w"

    processor = load_processor(args.adapter_path, args.min_pixels, args.max_pixels)
    processor.tokenizer.padding_side = "left"
    model = load_model(args.model_path, args.dtype)
    model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=False)
    model.to(args.device).eval()

    started = time.time()
    generated_records = 0
    total_unique_candidates = 0
    pending = [
        (index, record)
        for index, record in indexed_records
        if str(record["id"]) not in completed
    ]
    with output.open(mode, encoding="utf-8") as output_handle:
        for start in range(0, len(pending), args.batch_size):
            batch_items = pending[start : start + args.batch_size]
            prompts = [
                processor.apply_chat_template(
                    qwen_messages(record, include_answer=False),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for _, record in batch_items
            ]
            images_per_record = [load_images(record) for _, record in batch_items]
            flat_images = [image for images in images_per_record for image in images]
            inputs = processor(
                text=prompts,
                images=flat_images,
                padding=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(args.device) for key, value in inputs.items()}
            batch_seed = args.seed + batch_items[0][0]
            torch.manual_seed(batch_seed)
            torch.cuda.manual_seed_all(batch_seed)
            try:
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        repetition_penalty=args.repetition_penalty,
                        num_return_sequences=args.candidates_per_record,
                    )
                answer_tokens = generated[:, inputs["input_ids"].shape[1] :]
                raw_answers = processor.batch_decode(answer_tokens, skip_special_tokens=True)
            finally:
                for images in images_per_record:
                    for image in images:
                        image.close()
            expected_answers = len(batch_items) * args.candidates_per_record
            if len(raw_answers) != expected_answers:
                raise RuntimeError(
                    f"Expected {expected_answers} generations, got {len(raw_answers)}"
                )
            for batch_index, (original_index, record) in enumerate(batch_items):
                record_id = str(record["id"])
                begin = batch_index * args.candidates_per_record
                end = begin + args.candidates_per_record
                question = message_text(record, "user")
                candidates: list[str] = []
                for raw_answer in raw_answers[begin:end]:
                    candidate = format_challenge_answer(question, raw_answer.strip())
                    if candidate and candidate not in candidates:
                        candidates.append(candidate)
                item = {
                    "id": record_id,
                    "scene_id": str(record["scene_id"]),
                    "frame_id": str(record["frame_id"]),
                    "qa_index": int(record["qa_index"]),
                    "task": str(record["task"]),
                    "tag": record["tag"],
                    "source_record_index": original_index,
                    "shard_index": args.shard_index,
                    "batch_generation_seed": batch_seed,
                    "candidates": candidates,
                }
                output_handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                output_handle.flush()
                completed[record_id] = item
                generated_records += 1
                total_unique_candidates += len(candidates)
                print(
                    json.dumps(
                        {
                            "shard": args.shard_index,
                            "done": len(completed),
                            "total": len(indexed_records),
                            "id": record_id,
                            "batch_size": len(batch_items),
                            "unique_candidates": len(candidates),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    adapter_weights = Path(args.adapter_path) / "adapter_model.safetensors"
    report = {
        "schema_version": "drivelm-v037a-candidates-v1",
        "purpose": "local B10 train-only sampled candidates for offline preference construction",
        "train_manifest": str(Path(args.train_jsonl).resolve()),
        "train_manifest_sha256": sha256(args.train_jsonl),
        "adapter_path": str(Path(args.adapter_path).resolve()),
        "adapter_sha256": sha256(adapter_weights),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "records_expected": len(indexed_records),
        "records_complete": len(completed),
        "records_generated_this_run": generated_records,
        "unique_candidates_generated_this_run": total_unique_candidates,
        "generation": {
            "candidates_per_record": args.candidates_per_record,
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "seed": args.seed,
            "max_pixels": args.max_pixels,
        },
        "reference_answers_written": False,
        "external_api_used": False,
        "elapsed_seconds": time.time() - started,
    }
    report_path = output.with_suffix(output.suffix + ".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"completed": str(output), "report": str(report_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
