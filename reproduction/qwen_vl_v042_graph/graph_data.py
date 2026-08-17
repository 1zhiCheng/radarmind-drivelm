"""Frame-level multi-turn data utilities for DriveLM Graph-SFT."""

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


def read_graph_jsonl(path: str | Path, max_records: int = 0) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            missing = [value for value in record["images"].values() if not Path(value).is_file()]
            if missing:
                raise FileNotFoundError(f"{record['id']} references missing image {missing[0]}")
            if not record.get("nodes"):
                raise ValueError(f"{record['id']} contains no graph nodes")
            records.append(record)
            if max_records and len(records) >= max_records:
                break
    if not records:
        raise ValueError(f"No graph trajectories found in {path}")
    return records


class GraphTrajectoryDataset(torch.utils.data.Dataset):
    def __init__(self, records: list[dict[str, Any]]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def load_images(record: dict[str, Any]) -> list[Image.Image]:
    return [Image.open(record["images"][camera]).convert("RGB") for camera in CAMERA_ORDER]


def graph_messages(record: dict[str, Any], include_answers: bool = True) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": str(record["system"])}]}
    ]
    for index, node in enumerate(record["nodes"]):
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
        if include_answers:
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": str(node["answer"])}],
                }
            )
    return messages


def _find_subsequence(values: list[int], pattern: list[int], start: int = 0) -> int:
    limit = len(values) - len(pattern) + 1
    for index in range(start, max(start, limit)):
        if values[index : index + len(pattern)] == pattern:
            return index
    return -1


@dataclass
class GraphTrajectoryCollator:
    processor: Any
    max_length: int = 8192
    ignore_index: int = -100

    def __post_init__(self) -> None:
        tokenizer = self.processor.tokenizer
        self.assistant_header = tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )
        self.message_end = tokenizer.encode("<|im_end|>\n", add_special_tokens=False)
        if not self.assistant_header or not self.message_end:
            raise ValueError("Could not derive Qwen assistant boundary token IDs")
        self.last_stats: dict[str, Any] = {}

    def render(self, record: dict[str, Any]) -> str:
        return self.processor.apply_chat_template(
            graph_messages(record, include_answers=True),
            tokenize=False,
            add_generation_prompt=False,
        )

    def __call__(self, records: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        images_per_record = [load_images(record) for record in records]
        flat_images = [image for images in images_per_record for image in images]
        batch = self.processor(
            text=[self.render(record) for record in records],
            images=flat_images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = torch.full_like(batch["input_ids"], self.ignore_index)
        spans_per_row: list[int] = []
        supervised_per_row: list[int] = []
        for row_index, record in enumerate(records):
            ids = batch["input_ids"][row_index].tolist()
            attention = batch["attention_mask"][row_index].tolist()
            cursor = 0
            spans = 0
            for _node in record["nodes"]:
                header = _find_subsequence(ids, self.assistant_header, cursor)
                if header < 0:
                    break
                answer_start = header + len(self.assistant_header)
                end = _find_subsequence(ids, self.message_end, answer_start)
                if end < 0:
                    break
                answer_end = end + len(self.message_end)
                labels[row_index, answer_start:answer_end] = batch["input_ids"][
                    row_index, answer_start:answer_end
                ]
                cursor = answer_end
                spans += 1
            expected = len(record["nodes"])
            if spans != expected:
                raise ValueError(
                    f"{record['id']}: retained {spans}/{expected} assistant spans at "
                    f"max_length={self.max_length}; increase max length or reduce visual tokens"
                )
            labels[row_index, torch.tensor(attention, dtype=torch.bool).logical_not()] = self.ignore_index
            spans_per_row.append(spans)
            supervised_per_row.append(int((labels[row_index] != self.ignore_index).sum()))
        if not (labels != self.ignore_index).any():
            raise ValueError("No assistant tokens remain after graph masking")
        batch["labels"] = labels
        self.last_stats = {
            "assistant_spans_per_row": spans_per_row,
            "supervised_tokens_per_row": supervised_per_row,
            "sequence_lengths": [int(mask.sum()) for mask in batch["attention_mask"]],
        }
        return batch
