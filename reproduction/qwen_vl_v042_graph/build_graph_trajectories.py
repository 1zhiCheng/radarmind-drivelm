#!/usr/bin/env python3
"""Pack independent DriveLM QA rows into leakage-audited frame-level DAG trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TASK_ORDER = ("perception", "prediction", "planning", "behavior")
TASK_RANK = {task: index for index, task in enumerate(TASK_ORDER)}
SYSTEM_PROMPT = (
    "You are an autonomous-driving Graph-VQA agent. Inspect the six synchronized "
    "surround-view cameras once, then solve the graph trajectory in order: perception, "
    "prediction, planning, and behavior. Treat earlier answers as explicit working "
    "memory, preserve object IDs and coordinates exactly, and answer only the current "
    "node without revising previous nodes."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--dev-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def message(record: dict[str, Any], role: str) -> str:
    for item in record["messages"]:
        if item["role"] == role:
            return str(item["content"])
    raise KeyError(f"{record['id']} has no {role} message")


def pack(rows: list[dict[str, Any]], split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["scene_id"]), str(row["frame_id"]))].append(row)

    trajectories: list[dict[str, Any]] = []
    signature_counts: Counter[str] = Counter()
    edge_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    node_ids: set[str] = set()
    for (scene_id, frame_id), frame_rows in grouped.items():
        ordered = sorted(frame_rows, key=lambda item: int(item["qa_index"]))
        tasks = [str(item["task"]).lower() for item in ordered]
        unknown = sorted(set(tasks) - set(TASK_ORDER))
        if unknown:
            raise ValueError(f"{scene_id}/{frame_id}: unknown tasks {unknown}")
        if tasks != sorted(tasks, key=TASK_RANK.__getitem__):
            raise ValueError(f"{scene_id}/{frame_id}: task order is not monotonic: {tasks}")
        missing = [task for task in TASK_ORDER if task not in tasks]
        if missing:
            raise ValueError(f"{scene_id}/{frame_id}: incomplete trajectory, missing {missing}")
        if any(item["images"] != ordered[0]["images"] for item in ordered[1:]):
            raise ValueError(f"{scene_id}/{frame_id}: nodes do not share the same six images")

        nodes: list[dict[str, Any]] = []
        by_task: dict[str, list[str]] = defaultdict(list)
        for item in ordered:
            node_id = str(item["id"])
            if node_id in node_ids:
                raise ValueError(f"duplicate QA id: {node_id}")
            node_ids.add(node_id)
            task = str(item["task"]).lower()
            task_counts[task] += 1
            by_task[task].append(node_id)
            nodes.append(
                {
                    "id": node_id,
                    "qa_index": int(item["qa_index"]),
                    "task": task,
                    "tag": item.get("tag"),
                    "question": message(item, "user"),
                    "answer": message(item, "assistant"),
                }
            )

        edges: list[dict[str, str]] = []
        for source_task, target_task in zip(TASK_ORDER, TASK_ORDER[1:]):
            for source in by_task[source_task]:
                for target in by_task[target_task]:
                    edges.append({"source": source, "target": target})
                    edge_counts[f"{source_task}->{target_task}"] += 1

        signature = "/".join(f"{task}:{len(by_task[task])}" for task in TASK_ORDER)
        signature_counts[signature] += 1
        trajectories.append(
            {
                "id": f"{scene_id}_{frame_id}",
                "scene_id": scene_id,
                "frame_id": frame_id,
                "split": split,
                "system": SYSTEM_PROMPT,
                "images": ordered[0]["images"],
                "task_order": list(TASK_ORDER),
                "nodes": nodes,
                "edges": edges,
            }
        )

    report = {
        "split": split,
        "qa_rows": len(rows),
        "trajectories": len(trajectories),
        "scenes": len({row["scene_id"] for row in trajectories}),
        "task_counts": dict(task_counts),
        "node_count_min": min(len(row["nodes"]) for row in trajectories),
        "node_count_max": max(len(row["nodes"]) for row in trajectories),
        "signature_counts": dict(signature_counts),
        "stage_edge_counts": dict(edge_counts),
        "complete_four_stage_trajectories": len(trajectories),
    }
    return trajectories, report


def main() -> None:
    args = parse_args()
    train_rows = read_jsonl(args.train_jsonl)
    dev_rows = read_jsonl(args.dev_jsonl)
    train_scenes = {str(row["scene_id"]) for row in train_rows}
    dev_scenes = {str(row["scene_id"]) for row in dev_rows}
    overlap = sorted(train_scenes & dev_scenes)
    if overlap:
        raise ValueError(f"train/dev scene leakage: {overlap[:10]}")

    train, train_report = pack(train_rows, "train")
    dev, dev_report = pack(dev_rows, "dev")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "graph_train.jsonl"
    dev_path = args.output_dir / "graph_dev.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(dev_path, dev)
    report = {
        "schema_version": "radarmind-drivelm-v042-graph-trajectory-v1",
        "graph_contract": "frame-level stage DAG with explicit perception->prediction->planning->behavior edges",
        "teacher_forcing": "all assistant nodes are targets; no dev or hidden-val answers enter train",
        "train_dev_scene_overlap": 0,
        "source": {
            "train": str(args.train_jsonl.resolve()),
            "train_sha256": sha256(args.train_jsonl),
            "dev": str(args.dev_jsonl.resolve()),
            "dev_sha256": sha256(args.dev_jsonl),
        },
        "outputs": {
            "train": str(train_path.resolve()),
            "train_sha256": sha256(train_path),
            "dev": str(dev_path.resolve()),
            "dev_sha256": sha256(dev_path),
        },
        "train": train_report,
        "dev": dev_report,
    }
    (args.output_dir / "graph_manifest_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
