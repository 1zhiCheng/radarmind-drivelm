#!/usr/bin/env python3
"""Greedy six-camera inference using the exact VERL v0.40 trajectory prompt."""

from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from pathlib import Path

import torch
from datasets import Image as DatasetImage
from datasets import Sequence, load_dataset
from peft import PeftModel
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--dev-parquet", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--min-pixels", type=int, default=25088)
    parser.add_argument("--max-pixels", type=int, default=100352)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def structured_messages(row: dict) -> tuple[list[dict], list[Image.Image]]:
    messages = deepcopy(row["prompt"])
    paths = [str(item["path"]) for item in row["images"]]
    images = [Image.open(path).convert("RGB") for path in paths]
    offset = 0
    for message in messages:
        if not isinstance(message["content"], str):
            continue
        content = []
        for segment in filter(None, re.split(r"(<image>)", message["content"])):
            if segment == "<image>":
                if offset >= len(paths):
                    raise ValueError(f"too many image placeholders for {row['extra_info']['id']}")
                content.append({"type": "image", "image": paths[offset]})
                offset += 1
            else:
                content.append({"type": "text", "text": segment})
        message["content"] = content
    if offset != len(paths):
        raise ValueError(f"image placeholder mismatch for {row['extra_info']['id']}: {offset}/{len(paths)}")
    return messages, images


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    torch.manual_seed(42)
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        local_files_only=True,
        trust_remote_code=True,
    )
    processor.tokenizer.padding_side = "left"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=False)
    model.to(args.device).eval()
    dataset = load_dataset("parquet", data_files=args.dev_parquet, split="train")
    dataset = dataset.cast_column("images", Sequence(DatasetImage(decode=False)))
    output = Path(args.output_json)
    partial = output.with_suffix(output.suffix + ".partial.jsonl")
    predictions: dict[str, str] = {}
    if args.resume and partial.is_file():
        for line in partial.read_text().splitlines():
            if line.strip():
                item = json.loads(line)
                predictions[str(item["id"])] = str(item["answer"])
    pending = [index for index, row in enumerate(dataset) if str(row["extra_info"]["id"]) not in predictions]
    partial.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("a" if args.resume else "w", encoding="utf-8") as stream:
        for start in range(0, len(pending), args.batch_size):
            rows = [dataset[index] for index in pending[start : start + args.batch_size]]
            structured = [structured_messages(row) for row in rows]
            prompts = [
                processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                for messages, _ in structured
            ]
            images = [image for _, row_images in structured for image in row_images]
            inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt")
            inputs = {key: value.to(args.device) for key, value in inputs.items()}
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
            answers = processor.batch_decode(
                generated[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            for row, answer in zip(rows, answers, strict=True):
                record_id = str(row["extra_info"]["id"])
                predictions[record_id] = answer.strip()
                stream.write(json.dumps({"id": record_id, "answer": answer.strip()}, ensure_ascii=False) + "\n")
                stream.flush()
            for image in images:
                image.close()
            print(json.dumps({"done": len(predictions), "total": len(dataset)}), flush=True)
    ordered = [
        {"id": str(row["extra_info"]["id"]), "answer": predictions[str(row["extra_info"]["id"])]}
        for row in dataset
    ]
    output.write_text(json.dumps(ordered, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
