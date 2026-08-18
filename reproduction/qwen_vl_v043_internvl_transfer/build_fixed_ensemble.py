#!/usr/bin/env python3
"""Build leakage-free fixed-rule ensembles from retained DriveLM predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references-jsonl", type=Path, required=True)
    parser.add_argument("--baseline-json", type=Path, required=True)
    parser.add_argument("--grounding-json", type=Path, required=True)
    parser.add_argument("--mode", choices=("anchor", "anchor_tag3"), required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    return parser.parse_args()


def read_predictions(path: Path) -> dict[str, str]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    result = {str(row["id"]): str(row["answer"]) for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate prediction IDs in {path}")
    return result


def read_references(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def use_grounding(record: dict[str, Any], mode: str) -> bool:
    is_anchor = int(record["qa_index"]) == 0
    return is_anchor or (mode == "anchor_tag3" and 3 in record["tag"])


def main() -> None:
    args = parse_args()
    references = read_references(args.references_jsonl)
    baseline = read_predictions(args.baseline_json)
    grounding = read_predictions(args.grounding_json)
    reference_ids = {str(record["id"]) for record in references}
    for name, predictions in (("baseline", baseline), ("grounding", grounding)):
        if predictions.keys() != reference_ids:
            missing = sorted(reference_ids - predictions.keys())
            extra = sorted(predictions.keys() - reference_ids)
            raise ValueError(f"{name} coverage mismatch: missing={missing[:3]} extra={extra[:3]}")

    output: list[dict[str, str]] = []
    route_counts = {"baseline": 0, "grounding": 0}
    grounding_by_reason = {"anchor": 0, "tag3_non_anchor": 0}
    for record in references:
        record_id = str(record["id"])
        routed = use_grounding(record, args.mode)
        source = grounding if routed else baseline
        route_counts["grounding" if routed else "baseline"] += 1
        if routed:
            if int(record["qa_index"]) == 0:
                grounding_by_reason["anchor"] += 1
            else:
                grounding_by_reason["tag3_non_anchor"] += 1
        output.append({"id": record_id, "answer": source[record_id]})

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = {
        "schema_version": "drivelm-v043-fixed-ensemble-v1",
        "mode": args.mode,
        "selection_rule": "Graph-A for frame anchor" + (" and tag-3 QA" if args.mode == "anchor_tag3" else ""),
        "answer_or_reference_dependent_routing": False,
        "reference_count": len(references),
        "output_count": len(output),
        "route_counts": route_counts,
        "grounding_by_reason": grounding_by_reason,
        "sources": {
            "baseline": str(args.baseline_json.resolve()),
            "grounding": str(args.grounding_json.resolve()),
            "references": str(args.references_jsonl.resolve()),
        },
    }
    args.report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
