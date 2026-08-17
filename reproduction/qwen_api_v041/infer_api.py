#!/usr/bin/env python3
"""Resumable six-camera DriveLM inference through an OpenAI-compatible API."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI
from PIL import Image


CAMERA_ORDER = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)
DEFAULT_SYSTEM = (
    "You are an autonomous-driving perception and decision assistant. Analyze all six "
    "synchronized surround-view camera images. Answer the driving question accurately "
    "and concisely. Do not invent objects that are not visible. Return only the final "
    "answer, without analysis or reasoning traces."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="qwen3.8-max")
    parser.add_argument("--api-key-env", default="ALIYUN_TOKEN_PLAN_API_KEY")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-image-pixels", type=int, default=128 * 28 * 28)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def message_text(record: dict[str, Any], role: str) -> str:
    for message in record["messages"]:
        if message["role"] == role:
            return str(message["content"])
    if role == "system":
        return DEFAULT_SYSTEM
    raise KeyError(f"{record['id']} has no {role} message")


def encode_image(path: str, max_pixels: int, quality: int) -> str:
    with Image.open(path) as image:
        image = image.convert("RGB")
        pixels = image.width * image.height
        if pixels > max_pixels:
            scale = (max_pixels / pixels) ** 0.5
            size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
            image = image.resize(size, Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def build_messages(record: dict[str, Any], max_pixels: int, quality: int) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for camera in CAMERA_ORDER:
        path = str(record["images"][camera])
        if not Path(path).is_file():
            raise FileNotFoundError(f"{record['id']} missing {camera}: {path}")
        content.append({"type": "text", "text": f"[{camera}]"})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": encode_image(path, max_pixels, quality)},
            }
        )
    content.append({"type": "text", "text": message_text(record, "user")})
    return [
        {"role": "system", "content": message_text(record, "system") + " Return only the final answer."},
        {"role": "user", "content": content},
    ]


def normalize_answer(question: str, answer: str) -> str:
    answer = answer.strip()
    if "select the correct answer" in question.lower():
        import re

        match = re.search(r"(?:^|\b)([A-D])(?:[.):\s]|$)", answer, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return answer


def load_completed(path: Path) -> dict[str, dict[str, Any]]:
    partial = path.with_suffix(path.suffix + ".partial.jsonl")
    completed: dict[str, dict[str, Any]] = {}
    if not partial.is_file():
        return completed
    for line in partial.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            completed[str(row["id"])] = row
    return completed


def infer_one(
    record: dict[str, Any], client: OpenAI, args: argparse.Namespace
) -> dict[str, Any]:
    messages = build_messages(record, args.max_image_pixels, args.jpeg_quality)
    started = time.time()
    error = ""
    for attempt in range(args.max_retries):
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            raw = response.choices[0].message.content or ""
            answer = normalize_answer(message_text(record, "user"), raw)
            if not answer:
                raise ValueError("empty model answer")
            usage = response.usage
            return {
                "id": str(record["id"]),
                "answer": answer,
                "raw_answer": raw,
                "task": str(record["task"]),
                "model": args.model,
                "latency_seconds": round(time.time() - started, 3),
                "usage": {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                },
            }
        except Exception as exc:  # API failures are retried and recorded only after exhaustion.
            error = f"{type(exc).__name__}: {exc}"
            if attempt + 1 < args.max_retries:
                time.sleep(min(30.0, (2**attempt) + random.random()))
    raise RuntimeError(f"{record['id']} failed after {args.max_retries} attempts: {error}")


def main() -> None:
    args = parse_args()
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"missing API key environment variable: {args.api_key_env}")
    records = [json.loads(line) for line in args.input_jsonl.read_text(encoding="utf-8").splitlines() if line]
    if args.limit:
        records = records[: args.limit]
    expected_ids = {str(record["id"]) for record in records}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output_json.with_suffix(args.output_json.suffix + ".partial.jsonl")
    completed = load_completed(args.output_json) if args.resume else {}
    if set(completed) - expected_ids:
        raise ValueError("partial output contains IDs outside the requested input")
    pending = [record for record in records if str(record["id"]) not in completed]
    client = OpenAI(api_key=api_key, base_url=args.base_url.rstrip("/"), timeout=args.timeout)
    lock = threading.Lock()
    mode = "a" if args.resume else "w"
    with partial.open(mode, encoding="utf-8") as handle, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(infer_one, record, client, args): record for record in pending}
        for future in as_completed(futures):
            row = future.result()
            with lock:
                completed[row["id"]] = row
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                print(
                    json.dumps(
                        {
                            "completed": len(completed),
                            "total": len(records),
                            "id": row["id"],
                            "task": row["task"],
                            "latency_seconds": row["latency_seconds"],
                            "usage": row["usage"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    if set(completed) != expected_ids:
        raise RuntimeError(f"coverage mismatch: {len(completed)}/{len(expected_ids)}")
    ordered = [
        {"id": str(record["id"]), "answer": completed[str(record["id"])]["answer"]}
        for record in records
    ]
    args.output_json.write_text(json.dumps(ordered, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {
        "schema_version": "radarmind-drivelm-qwen38max-api-inference-v1",
        "model": args.model,
        "base_url": args.base_url,
        "input": str(args.input_jsonl.resolve()),
        "output": str(args.output_json.resolve()),
        "records": len(records),
        "coverage": 1.0,
        "workers": args.workers,
        "max_image_pixels": args.max_image_pixels,
        "jpeg_quality": args.jpeg_quality,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "api_key_persisted": False,
    }
    args.output_json.with_suffix(args.output_json.suffix + ".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
