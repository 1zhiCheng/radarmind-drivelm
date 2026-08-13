#!/usr/bin/env python3
"""Route frame anchors to one model and all downstream QA to another model."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references-jsonl", required=True)
    parser.add_argument("--anchor-predictions", required=True)
    parser.add_argument("--downstream-predictions", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--report-json", required=True)
    return parser.parse_args()


def load_predictions(path: str) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = json.loads(Path(path).read_text())
    if not isinstance(rows, list):
        raise TypeError(f"Predictions must be a JSON list: {path}")
    by_id = {str(row["id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError(f"Duplicate prediction IDs: {path}")
    return rows, by_id


def main() -> None:
    args = parse_args()
    with Path(args.references_jsonl).open(encoding="utf-8") as handle:
        references = [json.loads(line) for line in handle if line.strip()]

    _, anchor_by_id = load_predictions(args.anchor_predictions)
    downstream_rows, downstream_by_id = load_predictions(args.downstream_predictions)
    reference_ids = {str(row["id"]) for row in references}
    if set(anchor_by_id) != reference_ids or set(downstream_by_id) != reference_ids:
        raise ValueError("Both inputs must have exact reference-ID coverage")

    frames: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in references:
        frames[(str(row["scene_id"]), str(row["frame_id"]))].append(row)
    anchor_ids = {
        str(min(rows, key=lambda row: int(row["qa_index"]))["id"])
        for rows in frames.values()
    }

    output = []
    changed = 0
    for downstream_row in downstream_rows:
        row_id = str(downstream_row["id"])
        selected = anchor_by_id[row_id] if row_id in anchor_ids else downstream_by_id[row_id]
        output.append(selected)
        if selected.get("answer") != downstream_by_id[row_id].get("answer"):
            changed += 1

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    report = {
        "schema_version": "drivelm-v038b-anchor-routing-v1",
        "reference_count": len(references),
        "frame_count": len(frames),
        "anchor_count": len(anchor_ids),
        "downstream_count": len(references) - len(anchor_ids),
        "changed_anchor_answers": changed,
        "anchor_source": str(Path(args.anchor_predictions).resolve()),
        "downstream_source": str(Path(args.downstream_predictions).resolve()),
        "output": str(output_path.resolve()),
    }
    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
