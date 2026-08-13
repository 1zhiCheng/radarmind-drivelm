#!/usr/bin/env python3
"""Evaluate DriveLM graph-gating and coordinate metrics without an API judge."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "reproduction" / "drivelm_ds_eval"))
from metrics import graph_question_is_eligible, match_coordinates  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references-jsonl", required=True)
    parser.add_argument("--predictions-json", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def message(record: dict[str, Any], role: str) -> str:
    for item in record["messages"]:
        if item["role"] == role:
            return str(item["content"])
    raise KeyError(f"Missing {role} message for {record['id']}")


def main() -> None:
    args = parse_args()
    with Path(args.references_jsonl).open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    payload = json.loads(Path(args.predictions_json).read_text())
    predictions = {str(row["id"]): str(row["answer"]) for row in payload}
    reference_ids = {str(row["id"]) for row in records}
    missing = reference_ids - set(predictions)
    extra = set(predictions) - reference_ids

    frames: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        frames[(str(record["scene_id"]), str(record["frame_id"]))].append(record)
    eligible: list[dict[str, Any]] = []
    gated_out: list[str] = []
    anchor_rows: list[dict[str, float]] = []
    tag3_rows: list[dict[str, float]] = []
    for frame_records in frames.values():
        ordered = sorted(frame_records, key=lambda row: int(row["qa_index"]))
        first = ordered[0]
        first_id = str(first["id"])
        if first_id not in predictions:
            continue
        anchor_match = match_coordinates(
            predictions[first_id], message(first, "assistant")
        )
        anchor_rows.append({
            "precision": anchor_match.precision,
            "recall": anchor_match.recall,
            "f1": anchor_match.f1,
        })
        for index, record in enumerate(ordered):
            row_id = str(record["id"])
            if row_id not in predictions:
                continue
            if index and not graph_question_is_eligible(
                message(record, "user"), anchor_match.matched
            ):
                gated_out.append(row_id)
                continue
            eligible.append(record)
            if 3 in record.get("tag", []):
                match = match_coordinates(
                    predictions[row_id], message(record, "assistant")
                )
                tag3_rows.append({
                    "precision": match.precision,
                    "recall": match.recall,
                    "f1": match.f1,
                })

    def macro(rows: list[dict[str, float]]) -> dict[str, float | int]:
        return {
            "count": len(rows),
            "precision": sum(row["precision"] for row in rows) / max(len(rows), 1),
            "recall": sum(row["recall"] for row in rows) / max(len(rows), 1),
            "f1": sum(row["f1"] for row in rows) / max(len(rows), 1),
        }

    by_tag = Counter(str(tag) for row in eligible for tag in row.get("tag", []))
    report = {
        "schema_version": "drivelm-v038-structural-eval-v1",
        "reference_count": len(records),
        "prediction_count": len(predictions),
        "missing_count": len(missing),
        "extra_count": len(extra),
        "coverage": (len(reference_ids) - len(missing)) / max(len(reference_ids), 1),
        "eligible_count": len(eligible),
        "gated_out_count": len(gated_out),
        "eligible_by_tag": dict(sorted(by_tag.items())),
        "anchor_coordinate_macro": macro(anchor_rows),
        "tag3_coordinate_macro": macro(tag3_rows),
        "external_api_used": False,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if missing or extra:
        raise ValueError(f"Prediction coverage mismatch: missing={len(missing)}, extra={len(extra)}")


if __name__ == "__main__":
    main()
