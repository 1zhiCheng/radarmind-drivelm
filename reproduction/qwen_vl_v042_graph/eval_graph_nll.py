#!/usr/bin/env python3
"""Evaluate teacher-forced graph dev NLL for checkpoint screening only."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from peft import PeftModel
from torch.utils.data import DataLoader

from graph_data import GraphTrajectoryCollator, GraphTrajectoryDataset, read_graph_jsonl

import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "qwen_vl"))
from common import load_model, load_processor  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--graph-dev-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-trajectories", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--min-pixels", type=int, default=32 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=128 * 28 * 28)
    args = parser.parse_args()

    records = read_graph_jsonl(args.graph_dev_jsonl, args.max_trajectories)
    processor = load_processor(args.adapter_path, args.min_pixels, args.max_pixels)
    collator = GraphTrajectoryCollator(processor, max_length=args.max_length)
    loader = DataLoader(
        GraphTrajectoryDataset(records),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )
    model = load_model(args.model_path, args.dtype)
    model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=False)
    model.to(args.device).eval()
    weighted_loss = 0.0
    supervised_tokens = 0
    completed = 0
    with torch.inference_mode():
        for batch in loader:
            token_count = int((batch["labels"] != -100).sum())
            batch = {key: value.to(args.device) for key, value in batch.items()}
            loss = float(model(**batch).loss.detach().cpu())
            weighted_loss += loss * token_count
            supervised_tokens += token_count
            completed += int(batch["input_ids"].shape[0])
            if completed % 25 == 0 or completed == len(records):
                print(json.dumps({"completed": completed, "total": len(records)}), flush=True)
    mean_nll = weighted_loss / supervised_tokens
    report = {
        "schema_version": "radarmind-drivelm-v042-graph-nll-screen-v1",
        "scope": "teacher-forced labeled dev; checkpoint screen only, not promotion",
        "adapter": str(Path(args.adapter_path).resolve()),
        "trajectories": len(records),
        "qa_nodes": sum(len(record["nodes"]) for record in records),
        "supervised_tokens": supervised_tokens,
        "token_mean_nll": mean_nll,
        "perplexity": math.exp(min(mean_nll, 20.0)),
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
