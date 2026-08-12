#!/usr/bin/env python3
"""Generate DriveLM answers from the base Qwen-VL model or a LoRA adapter."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from peft import PeftModel

from common import load_images, load_model, load_processor, qwen_messages, read_jsonl


def format_challenge_answer(question: str, raw_answer: str) -> str:
    """Normalize constrained answers while leaving free-form answers unchanged."""
    if "please select the correct answer" in question.lower():
        match = re.match(r"\s*([A-D])(?:[.):\s]|$)", raw_answer, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return raw_answer.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--min-pixels", type=int, default=32 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=96 * 28 * 28)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.input_jsonl, args.max_samples)
    processor_path = args.adapter_path or args.model_path
    processor = load_processor(processor_path, args.min_pixels, args.max_pixels)
    # Decoder-only batched generation must continue after the last real token,
    # not after right-padding tokens.
    processor.tokenizer.padding_side = "left"
    model = load_model(args.model_path, args.dtype)
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=False)
    model.to(args.device).eval()
    output = Path(args.output_json)
    partial_output = output.with_suffix(output.suffix + ".partial.jsonl")
    predictions_by_id: dict[str, str] = {}
    if args.resume and partial_output.is_file():
        with partial_output.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    item = json.loads(line)
                    predictions_by_id[item["id"]] = item["answer"]
    pending_records = [record for record in records if record["id"] not in predictions_by_id]
    partial_output.parent.mkdir(parents=True, exist_ok=True)
    partial_mode = "a" if args.resume else "w"
    with partial_output.open(partial_mode, encoding="utf-8") as partial_handle:
        for start in range(0, len(pending_records), args.batch_size):
            batch_records = pending_records[start:start + args.batch_size]
            prompts = [
                processor.apply_chat_template(
                    qwen_messages(record, include_answer=False),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for record in batch_records
            ]
            images = [
                image
                for record in batch_records
                for image in load_images(record)
            ]
            inputs = processor(
                text=prompts,
                images=images,
                padding=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(args.device) for key, value in inputs.items()}
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            answer_tokens = generated[:, inputs["input_ids"].shape[1]:]
            raw_answers = processor.batch_decode(answer_tokens, skip_special_tokens=True)
            for record, raw_answer in zip(batch_records, raw_answers, strict=True):
                raw_answer = raw_answer.strip()
                question = next(x["content"] for x in record["messages"] if x["role"] == "user")
                answer = format_challenge_answer(str(question), raw_answer)
                predictions_by_id[record["id"]] = answer
                item = {"id": record["id"], "answer": answer}
                partial_handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                partial_handle.flush()
                print(json.dumps({"done": len(predictions_by_id), "total": len(records), "id": record["id"], "answer": answer, "raw_answer": raw_answer}, ensure_ascii=False), flush=True)
    predictions = [
        {"id": record["id"], "answer": predictions_by_id[record["id"]]}
        for record in records
        if record["id"] in predictions_by_id
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
