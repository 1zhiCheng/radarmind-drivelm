"""Shared multimodal preference utilities for DriveLM v0.37A."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image


HERE = Path(__file__).resolve().parent
COMMON_DIR = HERE.parent / "qwen_vl"
sys.path.insert(0, str(COMMON_DIR))
from common import CAMERA_ORDER  # noqa: E402


def read_preference_jsonl(path: str | Path, max_records: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = str(row["id"])
            if row_id in seen:
                raise ValueError(f"Duplicate id {row_id} in {path}:{line_number}")
            seen.add(row_id)
            missing = [
                row["images"].get(camera)
                for camera in CAMERA_ORDER
                if not row["images"].get(camera)
                or not Path(row["images"][camera]).is_file()
            ]
            if missing:
                raise FileNotFoundError(f"Preference {row_id} has missing camera input")
            for key in ("system", "question", "chosen", "rejected"):
                if not str(row.get(key, "")).strip():
                    raise ValueError(f"Preference {row_id} has empty {key}")
            rows.append(row)
            if max_records and len(rows) >= max_records:
                break
    if not rows:
        raise ValueError(f"No preference records found in {path}")
    return rows


def load_row_images(row: dict[str, Any]) -> list[Image.Image]:
    return [Image.open(row["images"][camera]).convert("RGB") for camera in CAMERA_ORDER]


def close_images(images: list[Image.Image]) -> None:
    for image in images:
        image.close()


def preference_messages(row: dict[str, Any], answer: str | None) -> list[dict[str, Any]]:
    user_content: list[dict[str, str]] = []
    for camera in CAMERA_ORDER:
        user_content.extend(
            [
                {"type": "text", "text": f"[{camera}]"},
                {"type": "image", "image": row["images"][camera]},
            ]
        )
    user_content.append({"type": "text", "text": str(row["question"])})
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": [{"type": "text", "text": str(row["system"])}],
        },
        {"role": "user", "content": user_content},
    ]
    if answer is not None:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": str(answer)}],
            }
        )
    return messages


def encode_answer(
    row: dict[str, Any],
    answer: str,
    processor: Any,
    images: list[Image.Image],
    max_length: int,
    device: str | torch.device,
) -> tuple[dict[str, torch.Tensor], int]:
    full_text = processor.apply_chat_template(
        preference_messages(row, answer),
        tokenize=False,
        add_generation_prompt=False,
    )
    prompt_text = processor.apply_chat_template(
        preference_messages(row, None),
        tokenize=False,
        add_generation_prompt=True,
    )
    batch = processor(
        text=[full_text],
        images=images,
        padding=False,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    prompt_batch = processor(
        text=[prompt_text],
        images=images,
        padding=False,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    labels = batch["input_ids"].clone()
    prompt_length = int(prompt_batch["input_ids"].shape[1])
    labels[:, :prompt_length] = -100
    labels[batch["attention_mask"] == 0] = -100
    response_tokens = int((labels != -100).sum().item())
    if response_tokens <= 0:
        raise ValueError(f"No answer tokens remain for {row['id']}")
    batch["labels"] = labels
    return {key: value.to(device) for key, value in batch.items()}, response_tokens


def sequence_score(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    response_tokens: int,
    normalization: str,
) -> torch.Tensor:
    outputs = model(**batch, use_cache=False)
    score = -outputs.loss.float() * response_tokens
    if normalization == "mean":
        score = score / response_tokens
    elif normalization != "sum":
        raise ValueError(f"Unsupported log-prob normalization: {normalization}")
    return score
