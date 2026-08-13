#!/usr/bin/env python3
"""Audit train-only important-object candidates under official graph gating."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "reproduction" / "drivelm_ds_eval"))
from metrics import coordinate_pairs, graph_question_is_eligible, match_coordinates  # noqa: E402


OBJECT_RE = re.compile(
    r"<[^,>]+,\s*[^,>]+,\s*\d+\.\d+,\s*\d+\.\d+>"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--candidate-jsonl", action="append", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-hard-jsonl", required=True)
    return parser.parse_args()


def message(record: dict[str, Any], role: str) -> str:
    for item in record["messages"]:
        if item["role"] == role:
            return str(item["content"])
    raise KeyError(f"Missing {role} message for {record['id']}")


def load_candidates(paths: list[str]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                row_id = str(row["id"])
                if row_id in result:
                    raise ValueError(f"Duplicate candidate id: {row_id}")
                result[row_id] = [str(value).strip() for value in row["candidates"]]
    return result


def main() -> None:
    args = parse_args()
    with Path(args.train_jsonl).open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    candidates = load_candidates(args.candidate_jsonl)
    frames: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        frames[(str(record["scene_id"]), str(record["frame_id"]))].append(record)

    hard_rows: list[dict[str, Any]] = []
    summary = Counter()
    f1_histogram = Counter()
    gate_loss_histogram = Counter()
    per_frame_gate_capacity = Counter()
    for frame_records in frames.values():
        ordered = sorted(frame_records, key=lambda row: int(row["qa_index"]))
        first = ordered[0]
        if int(first["qa_index"]) != 0 or 2 not in first.get("tag", []):
            continue
        row_id = str(first["id"])
        reference = message(first, "assistant")
        reference_pairs = coordinate_pairs(reference)
        downstream = ordered[1:]
        reference_eligible = sum(
            graph_question_is_eligible(message(row, "user"), tuple(reference_pairs))
            for row in downstream
        )
        per_frame_gate_capacity[str(reference_eligible)] += 1
        values = candidates.get(row_id, [])
        if not values:
            summary["important_object_without_candidates"] += 1
            continue
        summary["important_object_records"] += 1
        summary["candidate_answers"] += len(values)
        for candidate_index, candidate in enumerate(values):
            match = match_coordinates(candidate, reference)
            matched_graph = match.matched
            candidate_eligible = sum(
                graph_question_is_eligible(message(row, "user"), matched_graph)
                for row in downstream
            )
            gate_loss = reference_eligible - candidate_eligible
            complete_tuples = len(OBJECT_RE.findall(candidate))
            parsed_pairs = len(coordinate_pairs(candidate))
            likely_truncated = bool(candidate) and not candidate.rstrip().endswith((".", ">"))
            format_valid = complete_tuples > 0 and complete_tuples == parsed_pairs and not likely_truncated
            if likely_truncated:
                summary["likely_truncated"] += 1
            if not format_valid:
                summary["invalid_structured_format"] += 1
            if match.false_negatives:
                summary["coordinate_false_negative_candidates"] += 1
            if match.false_positives:
                summary["coordinate_false_positive_candidates"] += 1
            if gate_loss > 0:
                summary["candidates_losing_downstream_qa"] += 1
            if match.f1 < 0.999999 or not format_valid or gate_loss > 0:
                summary["hard_candidates"] += 1
                hard_rows.append({
                    "id": row_id,
                    "scene_id": str(first["scene_id"]),
                    "frame_id": str(first["frame_id"]),
                    "qa_index": 0,
                    "task": str(first["task"]),
                    "tag": first["tag"],
                    "images": first["images"],
                    "system": message(first, "system"),
                    "question": message(first, "user"),
                    "chosen": reference,
                    "rejected": candidate,
                    "candidate_index": candidate_index,
                    "coordinate": {
                        "reference_pairs": len(reference_pairs),
                        "candidate_pairs": parsed_pairs,
                        "complete_tuples": complete_tuples,
                        "tp": match.true_positives,
                        "fp": match.false_positives,
                        "fn": match.false_negatives,
                        "precision": match.precision,
                        "recall": match.recall,
                        "f1": match.f1,
                    },
                    "graph_gating": {
                        "downstream_records": len(downstream),
                        "reference_eligible": reference_eligible,
                        "candidate_eligible": candidate_eligible,
                        "eligible_loss": gate_loss,
                    },
                    "format": {
                        "valid_complete_tuples": format_valid,
                        "likely_truncated": likely_truncated,
                    },
                    "selection_score": (
                        2.0 * gate_loss
                        + match.false_negatives
                        + 0.5 * match.false_positives
                        + (0.5 if not format_valid else 0.0)
                    ),
                    "source": "B10 train-only candidates; no external API",
                })
            f1_histogram[f"{match.f1:.1f}"] += 1
            gate_loss_histogram[str(gate_loss)] += 1

    hard_rows.sort(
        key=lambda row: (-float(row["selection_score"]), row["id"], row["candidate_index"])
    )
    output_hard = Path(args.output_hard_jsonl)
    output_hard.parent.mkdir(parents=True, exist_ok=True)
    with output_hard.open("w", encoding="utf-8") as handle:
        for row in hard_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "schema_version": "drivelm-v038-graph-grounding-audit-v1",
        "scope": "train-only important-object candidates",
        "external_api_used": False,
        **dict(sorted(summary.items())),
        "coordinate_f1_histogram": dict(sorted(f1_histogram.items())),
        "downstream_eligible_loss_histogram": dict(
            sorted(gate_loss_histogram.items(), key=lambda item: int(item[0]))
        ),
        "reference_gate_capacity_by_frame": dict(
            sorted(per_frame_gate_capacity.items(), key=lambda item: int(item[0]))
        ),
        "hard_output": str(output_hard),
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
