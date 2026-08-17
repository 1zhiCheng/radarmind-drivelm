#!/usr/bin/env python3
"""Run predicted-context DriveLM graph rollouts without gold-answer leakage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "qwen_vl"))
from common import load_model, load_processor  # noqa: E402
from infer import format_challenge_answer  # noqa: E402
from graph_data import CAMERA_ORDER, load_images, read_graph_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--min-pixels", type=int, default=32 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=128 * 28 * 28)
    parser.add_argument("--max-trajectories", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def rollout_messages(
    record: dict[str, Any], predictions: dict[str, str], current_index: int
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": str(record["system"])}]}
    ]
    for index, node in enumerate(record["nodes"][: current_index + 1]):
        prefix = f"[GRAPH NODE {index + 1}/{len(record['nodes'])}] [{str(node['task']).upper()}]\n"
        if index == 0:
            content: list[dict[str, str]] = []
            for camera in CAMERA_ORDER:
                content.extend(
                    [
                        {"type": "text", "text": f"[{camera}]"},
                        {"type": "image", "image": record["images"][camera]},
                    ]
                )
            content.append({"type": "text", "text": prefix + str(node["question"])})
        else:
            content = [{"type": "text", "text": prefix + str(node["question"])}]
        messages.append({"role": "user", "content": content})
        if index < current_index:
            node_id = str(node["id"])
            if node_id not in predictions:
                raise ValueError(f"Missing upstream predicted answer {node_id}")
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": predictions[node_id]}],
                }
            )
    return messages


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard index/count")
    all_records = read_graph_jsonl(args.input_jsonl)
    records = [row for index, row in enumerate(all_records) if index % args.num_shards == args.shard_index]
    if args.max_trajectories:
        records = records[: args.max_trajectories]
    processor = load_processor(args.adapter_path, args.min_pixels, args.max_pixels)
    processor.tokenizer.padding_side = "left"
    model = load_model(args.model_path, args.dtype)
    model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=False)
    model.to(args.device).eval()

    output = Path(args.output_json)
    partial = output.with_suffix(output.suffix + ".partial.jsonl")
    predictions: dict[str, str] = {}
    if args.resume and partial.is_file():
        with partial.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    predictions[str(row["id"])] = str(row["answer"])
    partial.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    total_nodes = sum(len(record["nodes"]) for record in records)
    with partial.open(mode, encoding="utf-8") as handle:
        max_depth = max(len(record["nodes"]) for record in records)
        for depth in range(max_depth):
            active = [
                record
                for record in records
                if depth < len(record["nodes"])
                and str(record["nodes"][depth]["id"]) not in predictions
            ]
            for start in range(0, len(active), args.batch_size):
                batch_records = active[start : start + args.batch_size]
                prompts = [
                    processor.apply_chat_template(
                        rollout_messages(record, predictions, depth),
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for record in batch_records
                ]
                images = [image for record in batch_records for image in load_images(record)]
                inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt")
                inputs = {key: value.to(args.device) for key, value in inputs.items()}
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                    )
                answer_tokens = generated[:, inputs["input_ids"].shape[1] :]
                raw_answers = processor.batch_decode(answer_tokens, skip_special_tokens=True)
                for record, raw_answer in zip(batch_records, raw_answers, strict=True):
                    node = record["nodes"][depth]
                    answer = format_challenge_answer(str(node["question"]), raw_answer.strip())
                    node_id = str(node["id"])
                    predictions[node_id] = answer
                    row = {
                        "id": node_id,
                        "answer": answer,
                        "task": node["task"],
                        "trajectory_id": record["id"],
                        "node_index": depth,
                    }
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                print(
                    json.dumps(
                        {
                            "shard": f"{args.shard_index}/{args.num_shards}",
                            "depth": depth,
                            "completed": len(predictions),
                            "total": total_nodes,
                        }
                    ),
                    flush=True,
                )

    expected = [str(node["id"]) for record in records for node in record["nodes"]]
    missing = [node_id for node_id in expected if node_id not in predictions]
    if missing:
        raise RuntimeError(f"Graph rollout incomplete: {len(missing)} missing, first={missing[0]}")
    output.write_text(
        json.dumps(
            [{"id": node_id, "answer": predictions[node_id]} for node_id in expected],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "predictions": len(expected), "coverage": 1.0}))


if __name__ == "__main__":
    main()
