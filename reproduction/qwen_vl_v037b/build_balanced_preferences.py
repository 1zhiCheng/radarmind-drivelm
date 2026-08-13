#!/usr/bin/env python3
"""Build a deterministic task-balanced reference-scored preference manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


TASKS = ("perception", "prediction", "planning", "behavior")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--per-task", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    grouped: dict[str, list[dict]] = defaultdict(list)
    seen: set[str] = set()
    with Path(args.input_jsonl).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = str(row["id"])
            if row_id in seen:
                raise ValueError(f"Duplicate id {row_id} at line {line_number}")
            seen.add(row_id)
            task = str(row["task"])
            if task not in TASKS:
                raise ValueError(f"Unexpected task {task!r}")
            for key in ("reference_chosen_logp", "reference_rejected_logp"):
                if key not in row:
                    raise ValueError(f"Record {row_id} is missing {key}")
            grouped[task].append(row)
    missing_tasks = [task for task in TASKS if not grouped[task]]
    if missing_tasks:
        raise ValueError(f"Missing tasks: {missing_tasks}")
    available = {task: len(grouped[task]) for task in TASKS}
    target = args.per_task or min(available.values())
    if target <= 0 or any(count < target for count in available.values()):
        raise ValueError(f"Invalid per-task target {target}; available={available}")

    selected: dict[str, list[dict]] = {}
    for task_index, task in enumerate(TASKS):
        task_rows = list(grouped[task])
        random.Random(args.seed + task_index).shuffle(task_rows)
        selected[task] = task_rows[:target]
    balanced: list[dict] = []
    for index in range(target):
        for task in TASKS:
            row = dict(selected[task][index])
            row["v037b_sampling"] = {
                "schema_version": "drivelm-v037b-balanced-v1",
                "strategy": "equal unique pairs per task, round-robin order",
                "task": task,
                "task_index": index,
                "seed": args.seed,
            }
            balanced.append(row)

    output = Path(args.output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in balanced:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "schema_version": "drivelm-v037b-balanced-v1",
        "input_records": sum(available.values()),
        "input_by_task": available,
        "per_task": target,
        "output_records": len(balanced),
        "output_by_task": {task: target for task in TASKS},
        "unique_output_ids": len({str(row["id"]) for row in balanced}),
        "round_robin_order": list(TASKS),
        "replacement": False,
        "seed": args.seed,
        "input_sha256": sha256(args.input_jsonl),
        "output_sha256": sha256(output),
    }
    if report["unique_output_ids"] != report["output_records"]:
        raise AssertionError("Balanced preference output contains duplicate IDs")
    report_path = output.with_suffix(output.suffix + ".report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
