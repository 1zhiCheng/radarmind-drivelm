#!/usr/bin/env python3
"""Build the exact graph-gating intersection for a paired model audit."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


METRIC_DIR = Path(__file__).resolve().parents[1] / "drivelm_ds_eval"
sys.path.insert(0, str(METRIC_DIR))
from metrics import graph_question_is_eligible, match_coordinates  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references-jsonl", required=True)
    parser.add_argument("--baseline-predictions", required=True)
    parser.add_argument("--candidate-predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def read_jsonl(path: str) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_predictions(path: str) -> tuple[dict[str, str], list[dict[str, str]]]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, list):
        raise TypeError(f"Predictions must be a list: {path}")
    rows = [{"id": str(row["id"]), "answer": str(row["answer"])} for row in payload]
    return {row["id"]: row["answer"] for row in rows}, rows


def message(record: dict[str, Any], role: str) -> str:
    for item in record["messages"]:
        if item["role"] == role:
            return str(item["content"])
    raise KeyError(f"Missing {role} message for {record['id']}")


def eligible_ids(
    records: list[dict[str, Any]], predictions: dict[str, str]
) -> set[str]:
    frames: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        frames[(str(record["scene_id"]), str(record["frame_id"]))].append(record)
    eligible: set[str] = set()
    for frame_records in frames.values():
        ordered = sorted(frame_records, key=lambda row: int(row["qa_index"]))
        first = ordered[0]
        matched = match_coordinates(
            predictions[str(first["id"])], message(first, "assistant")
        ).matched
        for index, record in enumerate(ordered):
            if index and not graph_question_is_eligible(message(record, "user"), matched):
                continue
            eligible.add(str(record["id"]))
    return eligible


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.references_jsonl)
    baseline, baseline_rows = load_predictions(args.baseline_predictions)
    candidate, candidate_rows = load_predictions(args.candidate_predictions)
    reference_ids = {str(row["id"]) for row in records}
    if set(baseline) != reference_ids or set(candidate) != reference_ids:
        raise ValueError("Both prediction files must have exactly 100% reference coverage")

    baseline_eligible = eligible_ids(records, baseline)
    candidate_eligible = eligible_ids(records, candidate)
    common = baseline_eligible & candidate_eligible
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected_records = [row for row in records if str(row["id"]) in common]

    with (output / "common_references.jsonl").open("w", encoding="utf-8") as handle:
        for row in selected_records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    for name, rows in (("baseline", baseline_rows), ("candidate", candidate_rows)):
        selected = [row for row in rows if row["id"] in common]
        (output / f"{name}_predictions.json").write_text(
            json.dumps(selected, ensure_ascii=False, indent=2) + "\n"
        )

    common_by_tag = Counter(
        str(tag) for row in selected_records for tag in row.get("tag", [])
    )
    report = {
        "schema_version": "drivelm-v037b-common-gating-audit-v1",
        "full_reference_count": len(records),
        "baseline_eligible": len(baseline_eligible),
        "candidate_eligible": len(candidate_eligible),
        "common_eligible": len(common),
        "baseline_only": len(baseline_eligible - candidate_eligible),
        "candidate_only": len(candidate_eligible - baseline_eligible),
        "common_by_tag": dict(sorted(common_by_tag.items())),
        "common_reference_count": len(selected_records),
        "baseline_common_predictions": sum(row["id"] in common for row in baseline_rows),
        "candidate_common_predictions": sum(row["id"] in common for row in candidate_rows),
    }
    (output / "common_subset_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
