"""Shared six-camera Qwen-VL data utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image


CAMERA_ORDER = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)


def read_jsonl(path: str | Path, max_records: int = 0) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            missing = [p for p in record["images"].values() if not Path(p).is_file()]
            if missing:
                raise FileNotFoundError(f"Record {record['id']} references missing image {missing[0]}")
            records.append(record)
            if max_records and len(records) >= max_records:
                break
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def message_text(record: dict[str, Any], role: str) -> str:
    for message in record["messages"]:
        if message["role"] == role:
            return str(message["content"])
    raise KeyError(f"{record['id']} has no {role} message")


def qwen_messages(record: dict[str, Any], include_answer: bool) -> list[dict[str, Any]]:
    user_content: list[dict[str, str]] = []
    for camera in CAMERA_ORDER:
        user_content.extend(
            [
                {"type": "text", "text": f"[{camera}]"},
                {"type": "image", "image": record["images"][camera]},
            ]
        )
    user_content.append({"type": "text", "text": message_text(record, "user")})
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": [{"type": "text", "text": message_text(record, "system")}],
        },
        {"role": "user", "content": user_content},
    ]
    if include_answer:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": message_text(record, "assistant")}],
            }
        )
    return messages


def load_images(record: dict[str, Any]) -> list[Image.Image]:
    return [Image.open(record["images"][camera]).convert("RGB") for camera in CAMERA_ORDER]


class DriveLMDataset(torch.utils.data.Dataset):
    def __init__(self, records: list[dict[str, Any]]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


@dataclass
class SixCameraCollator:
    processor: Any
    max_length: int = 2048
    ignore_index: int = -100

    def render(self, record: dict[str, Any], include_answer: bool) -> str:
        return self.processor.apply_chat_template(
            qwen_messages(record, include_answer),
            tokenize=False,
            add_generation_prompt=not include_answer,
        )

    def __call__(self, records: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        images_per_record = [load_images(record) for record in records]
        flat_images = [image for images in images_per_record for image in images]
        full_texts = [self.render(record, True) for record in records]
        prompt_texts = [self.render(record, False) for record in records]
        batch = self.processor(
            text=full_texts,
            images=flat_images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()
        for row, (prompt, images) in enumerate(zip(prompt_texts, images_per_record, strict=True)):
            prompt_batch = self.processor(
                text=[prompt],
                images=images,
                padding=False,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            prompt_len = int(prompt_batch["input_ids"].shape[1])
            labels[row, :prompt_len] = self.ignore_index
        labels[batch["attention_mask"] == 0] = self.ignore_index
        if not (labels != self.ignore_index).any():
            raise ValueError("No assistant tokens remain; increase --max-length or lower --max-pixels")
        batch["labels"] = labels
        return batch


def load_processor(model_path: str, min_pixels: int, max_pixels: int) -> Any:
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(
        model_path,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        trust_remote_code=True,
        local_files_only=True,
    )


def load_model(model_path: str, dtype: str = "bf16") -> torch.nn.Module:
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
    except ImportError:
        from transformers import AutoModelForImageTextToText as ModelClass
    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
    return ModelClass.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        local_files_only=True,
    )
