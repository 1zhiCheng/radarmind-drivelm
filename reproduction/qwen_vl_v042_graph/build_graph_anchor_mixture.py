#!/usr/bin/env python3
"""Build the deterministic v0.42B 1:1 Graph / task-balanced anchor manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

TASKS = ("perception", "prediction", "planning", "behavior")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-jsonl", required=True)
    parser.add_argument("--independent-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--anchors-per-task", type=int, default=901)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_jsonl(path: str) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def message(row: dict[str, Any], role: str) -> str:
    for item in row["messages"]:
        if item["role"] == role:
            return str(item["content"])
    raise KeyError(f"{row['id']} has no {role} message")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.anchors_per_task <= 0:
        raise ValueError("--anchors-per-task must be positive")
    graphs = read_jsonl(args.graph_jsonl)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(args.independent_jsonl):
        groups[str(row["task"])].append(row)
    rng = random.Random(args.seed)
    anchors: list[dict[str, Any]] = []
    for task in TASKS:
        rows = sorted(groups[task], key=lambda row: str(row["id"]))
        if len(rows) < args.anchors_per_task:
            raise ValueError(f"{task} only has {len(rows)} rows")
        rng.shuffle(rows)
        for row in rows[: args.anchors_per_task]:
            anchors.append(
                {
                    "id": "anchor::" + str(row["id"]),
                    "scene_id": str(row["scene_id"]),
                    "frame_id": str(row["frame_id"]),
                    "images": row["images"],
                    "system": message(row, "system"),
                    "nodes": [
                        {
                            "id": str(row["id"]),
                            "qa_index": int(row["qa_index"]),
                            "task": task,
                            "tag": row.get("tag", []),
                            "question": message(row, "user"),
                            "answer": message(row, "assistant"),
                        }
                    ],
                    "edges": [],
                    "training_mode": "task_balanced_single_node_ce_anchor",
                }
            )
    for row in graphs:
        row["training_mode"] = "full_four_stage_graph"
    combined = graphs + anchors
    rng.shuffle(combined)
    output = Path(args.output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in combined:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "schema_version": "radarmind-drivelm-v042b-graph-anchor-manifest-v1",
        "seed": args.seed,
        "full_graph_trajectories": len(graphs),
        "single_node_anchors": len(anchors),
        "anchor_by_task": dict(Counter(row["nodes"][0]["task"] for row in anchors)),
        "combined_records": len(combined),
        "mix_ratio_graph": len(graphs) / len(combined),
        "output": str(output.resolve()),
        "sha256": sha256(output),
    }
    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
