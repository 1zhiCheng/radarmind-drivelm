#!/usr/bin/env python3
"""Build balanced v0.38 preferences with fresh anchor/tag-3 grounding negatives."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "reproduction" / "drivelm_ds_eval"))
from metrics import coordinate_pairs, graph_question_is_eligible, match_coordinates  # noqa: E402


TASKS = ("perception", "prediction", "planning", "behavior")
TUPLE_RE = re.compile(r"<[^,>]+,\s*[^,>]+,\s*\d+\.\d+,\s*\d+\.\d+>")
TOKEN_RE = re.compile(r"[A-Za-z0-9_.-]+|<|>")
REFERENCE_FIELDS = (
    "reference_policy", "reference_logp_normalization", "reference_chosen_logp",
    "reference_rejected_logp", "reference_chosen_tokens", "reference_rejected_tokens",
    "v037b_sampling",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--dev-jsonl", required=True)
    parser.add_argument("--anchor-candidate-jsonl", action="append", required=True)
    parser.add_argument("--tag3-candidate-jsonl", action="append", required=True)
    parser.add_argument("--replay-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--per-task", type=int, default=1026)
    parser.add_argument("--max-coordinate-f1", type=float, default=0.75)
    parser.add_argument("--min-length-ratio", type=float, default=0.60)
    parser.add_argument("--max-length-ratio", type=float, default=1.45)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def message(record: dict[str, Any], role: str) -> str:
    for item in record["messages"]:
        if item["role"] == role:
            return str(item["content"])
    raise KeyError(f"Missing {role} message for {record['id']}")


def load_candidates(paths: list[str]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for path in paths:
        for row in read_jsonl(path):
            row_id = str(row["id"])
            if row_id in result:
                raise ValueError(f"Duplicate candidate id {row_id}")
            result[row_id] = [str(value).strip() for value in row["candidates"]]
    return result


def structured_complete(text: str) -> bool:
    tuples = TUPLE_RE.findall(text)
    return (
        bool(tuples)
        and len(tuples) == len(coordinate_pairs(text))
        and text.count("<") == text.count(">") == len(tuples)
        and text.rstrip().endswith((".", ">"))
    )


def token_length(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def clean_reference_fields(row: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(row)
    for key in REFERENCE_FIELDS:
        cleaned.pop(key, None)
    return cleaned


def main() -> None:
    args = parse_args()
    train = read_jsonl(args.train_jsonl)
    dev = read_jsonl(args.dev_jsonl)
    train_by_id = {str(row["id"]): row for row in train}
    dev_ids = {str(row["id"]) for row in dev}
    train_scenes = {str(row["scene_id"]) for row in train}
    dev_scenes = {str(row["scene_id"]) for row in dev}
    if train_scenes & dev_scenes:
        raise ValueError("Train/dev scene leakage detected")

    frames: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in train:
        frames[(str(row["scene_id"]), str(row["frame_id"]))].append(row)
    frame_by_id = {
        str(row["id"]): sorted(frame, key=lambda item: int(item["qa_index"]))
        for frame in frames.values() for row in frame
    }

    fresh_candidates = load_candidates(
        args.anchor_candidate_jsonl + args.tag3_candidate_jsonl
    )
    unexpected = set(fresh_candidates) - set(train_by_id)
    if unexpected:
        raise ValueError(f"Fresh candidates contain {len(unexpected)} non-train IDs")
    fresh_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    exclusion = Counter()
    valid_candidates = 0
    for row_id, values in fresh_candidates.items():
        record = train_by_id[row_id]
        chosen = message(record, "assistant").strip()
        chosen_tokens = token_length(chosen)
        accepted: list[tuple[float, dict[str, Any]]] = []
        for candidate_index, rejected in enumerate(values):
            if not rejected or rejected == chosen:
                exclusion["empty_or_equal"] += 1
                continue
            if not structured_complete(rejected):
                exclusion["incomplete_structure"] += 1
                continue
            length_ratio = token_length(rejected) / max(chosen_tokens, 1)
            if not args.min_length_ratio <= length_ratio <= args.max_length_ratio:
                exclusion["length_ratio"] += 1
                continue
            match = match_coordinates(rejected, chosen)
            frame = frame_by_id[row_id]
            gate_loss = 0
            if int(record["qa_index"]) == 0:
                reference_pairs = tuple(coordinate_pairs(chosen))
                reference_eligible = sum(
                    graph_question_is_eligible(message(item, "user"), reference_pairs)
                    for item in frame[1:]
                )
                candidate_eligible = sum(
                    graph_question_is_eligible(message(item, "user"), match.matched)
                    for item in frame[1:]
                )
                gate_loss = reference_eligible - candidate_eligible
            if match.f1 > args.max_coordinate_f1 and gate_loss <= 0:
                exclusion["not_hard_enough"] += 1
                continue
            valid_candidates += 1
            score = (
                2.0 * gate_loss
                + 2.0 * (1.0 - match.f1)
                + 0.25 * match.false_negatives
                + 0.10 * match.false_positives
            )
            accepted.append((score, {
                "id": row_id,
                "scene_id": str(record["scene_id"]),
                "frame_id": str(record["frame_id"]),
                "qa_index": int(record["qa_index"]),
                "task": str(record["task"]),
                "tag": record["tag"],
                "images": record["images"],
                "system": message(record, "system"),
                "question": message(record, "user"),
                "chosen": chosen,
                "rejected": rejected,
                "selection_reason": "v038_complete_grounding_error",
                "selection_score": score,
                "selection_details": {
                    "candidate_index": candidate_index,
                    "coordinate_f1": match.f1,
                    "coordinate_tp": match.true_positives,
                    "coordinate_fp": match.false_positives,
                    "coordinate_fn": match.false_negatives,
                    "downstream_eligible_loss": gate_loss,
                    "length_ratio": length_ratio,
                    "structured_complete": True,
                },
                "policy_source": "v0.37B checkpoint-75",
                "external_api_used": False,
            }))
        if accepted:
            accepted.sort(key=lambda item: (-item[0], item[1]["rejected"]))
            fresh_by_task[str(record["task"])].append(accepted[0][1])

    replay_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(args.replay_jsonl):
        replay_by_task[str(row["task"])].append(clean_reference_fields(row))
    selected: dict[str, list[dict[str, Any]]] = {}
    source_counts: dict[str, dict[str, int]] = {}
    for task_index, task in enumerate(TASKS):
        fresh = list(fresh_by_task[task])
        fresh.sort(key=lambda row: (-float(row["selection_score"]), row["id"]))
        task_selected = fresh[: args.per_task]
        used = {str(row["id"]) for row in task_selected}
        replay = [row for row in replay_by_task[task] if str(row["id"]) not in used]
        random.Random(args.seed + task_index).shuffle(replay)
        replay_needed = args.per_task - len(task_selected)
        if replay_needed > len(replay):
            raise ValueError(
                f"Insufficient {task} pairs: fresh={len(task_selected)}, replay={len(replay)}"
            )
        task_selected.extend(replay[:replay_needed])
        random.Random(args.seed + 100 + task_index).shuffle(task_selected)
        selected[task] = task_selected
        source_counts[task] = {
            "fresh_grounding": len(task_selected) - replay_needed,
            "v037b_replay": replay_needed,
        }

    balanced: list[dict[str, Any]] = []
    for index in range(args.per_task):
        for task in TASKS:
            row = clean_reference_fields(selected[task][index])
            row["v038_sampling"] = {
                "schema_version": "drivelm-v038-grounding-balanced-v1",
                "strategy": "fresh complete grounding pairs with task-balanced replay",
                "task": task,
                "task_index": index,
                "seed": args.seed,
            }
            balanced.append(row)
    output_ids = [str(row["id"]) for row in balanced]
    if len(output_ids) != len(set(output_ids)):
        raise AssertionError("v0.38 preference manifest contains duplicate IDs")
    if set(output_ids) & dev_ids:
        raise AssertionError("v0.38 preference manifest contains dev IDs")

    selected_fresh_stats: dict[str, dict[str, Any]] = {}
    for task in TASKS:
        fresh_rows = [
            row for row in selected[task]
            if row.get("selection_reason") == "v038_complete_grounding_error"
        ]
        f1_histogram = Counter(
            f"{float(row['selection_details']['coordinate_f1']):.1f}"
            for row in fresh_rows
        )
        gate_histogram = Counter(
            str(int(row["selection_details"]["downstream_eligible_loss"]))
            for row in fresh_rows
        )
        selected_fresh_stats[task] = {
            "count": len(fresh_rows),
            "coordinate_f1_histogram": dict(sorted(f1_histogram.items())),
            "downstream_eligible_loss_histogram": dict(
                sorted(gate_histogram.items(), key=lambda item: int(item[0]))
            ),
            "mean_coordinate_f1": (
                sum(float(row["selection_details"]["coordinate_f1"]) for row in fresh_rows)
                / len(fresh_rows) if fresh_rows else None
            ),
            "mean_length_ratio": (
                sum(float(row["selection_details"]["length_ratio"]) for row in fresh_rows)
                / len(fresh_rows) if fresh_rows else None
            ),
        }

    output = Path(args.output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in balanced:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "schema_version": "drivelm-v038-grounding-balanced-v1",
        "train_records": len(train),
        "dev_records": len(dev),
        "train_dev_scene_overlap": 0,
        "fresh_candidate_records": len(fresh_candidates),
        "valid_complete_hard_candidates": valid_candidates,
        "fresh_pairs_by_task": {
            task: len(rows) for task, rows in sorted(fresh_by_task.items())
        },
        "candidate_exclusion": dict(sorted(exclusion.items())),
        "per_task": args.per_task,
        "output_records": len(balanced),
        "unique_output_ids": len(set(output_ids)),
        "dev_ids_in_output": len(set(output_ids) & dev_ids),
        "source_counts": source_counts,
        "selected_fresh_stats": selected_fresh_stats,
        "round_robin_order": list(TASKS),
        "reference_logp_present": False,
        "external_api_used": False,
        "input_sha256": {
            "train": sha256(args.train_jsonl),
            "dev": sha256(args.dev_jsonl),
            "replay": sha256(args.replay_jsonl),
        },
        "output_sha256": sha256(output),
    }
    output.with_suffix(output.suffix + ".report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
