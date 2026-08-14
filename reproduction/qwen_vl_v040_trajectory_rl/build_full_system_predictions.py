#!/usr/bin/env python3
"""Replace only MoL Planning answers with trajectory-RL predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references-jsonl", required=True)
    parser.add_argument("--mol-predictions", required=True)
    parser.add_argument("--planning-predictions", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--report-json", required=True)
    return parser.parse_args()


def load_predictions(path: str) -> list[dict[str, str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError(f"prediction payload must be a list: {path}")
    rows = [{"id": str(row["id"]), "answer": str(row["answer"])} for row in payload]
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"duplicate prediction IDs: {path}")
    return rows


def main() -> None:
    args = parse_args()
    references = [
        json.loads(line)
        for line in Path(args.references_jsonl).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    reference_by_id = {str(row["id"]): row for row in references}
    mol_rows = load_predictions(args.mol_predictions)
    planning_rows = load_predictions(args.planning_predictions)
    mol = {row["id"]: row["answer"] for row in mol_rows}
    planning = {row["id"]: row["answer"] for row in planning_rows}
    reference_ids = set(reference_by_id)
    planning_ids = {
        row_id
        for row_id, row in reference_by_id.items()
        if str(row["task"]) == "planning"
    }
    if set(mol) != reference_ids:
        raise ValueError("MoL predictions do not have exact full-dev coverage")
    if set(planning) != planning_ids:
        raise ValueError("RL predictions do not exactly cover Planning IDs")
    merged = [
        {
            "id": str(row["id"]),
            "answer": planning.get(str(row["id"]), mol[str(row["id"])]),
        }
        for row in references
    ]
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n")
    report = {
        "schema_version": "radarmind-v040-full-system-merge-v1",
        "reference_count": len(references),
        "prediction_count": len(merged),
        "coverage": 1.0,
        "planning_replaced": len(planning),
        "non_planning_frozen": len(references) - len(planning),
        "routing": "v0.39B step-700 experts; replace Planning only with trajectory RL",
    }
    Path(args.report_json).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
