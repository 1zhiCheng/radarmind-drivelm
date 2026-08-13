#!/usr/bin/env python3
"""Build an oracle-memory probe for newly eligible DriveLM planning QA.

This is a dev-only diagnostic. Its output must never be used for training or
reported as a promotion score because it injects the reference frame anchor.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "reproduction" / "drivelm_ds_eval"))
from metrics import graph_question_is_eligible, match_coordinates  # noqa: E402

TUPLE_RE = re.compile(r"<[^,>]+,\s*[^,>]+,\s*\d+\.\d+,\s*\d+\.\d+>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references-jsonl", required=True)
    parser.add_argument("--baseline-predictions", required=True)
    parser.add_argument("--candidate-predictions", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--baseline-output-json", required=True)
    parser.add_argument("--judge-references-jsonl", required=True)
    return parser.parse_args()


def text(record: dict[str, Any], role: str) -> str:
    return str(next(item["content"] for item in record["messages"] if item["role"] == role))


def predictions(path: str) -> dict[str, str]:
    rows = json.loads(Path(path).read_text())
    result = {str(row["id"]): str(row["answer"]) for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate prediction IDs in {path}")
    return result


def eligible_ids(records: list[dict[str, Any]], answers: dict[str, str]) -> set[str]:
    frames: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        frames[(str(record["scene_id"]), str(record["frame_id"]))].append(record)
    result: set[str] = set()
    for frame in frames.values():
        ordered = sorted(frame, key=lambda row: int(row["qa_index"]))
        anchor = ordered[0]
        matched = match_coordinates(answers[str(anchor["id"])], text(anchor, "assistant")).matched
        for index, record in enumerate(ordered):
            if index == 0 or graph_question_is_eligible(text(record, "user"), matched):
                result.add(str(record["id"]))
    return result


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    reference_path = Path(args.references_jsonl)
    records = [json.loads(line) for line in reference_path.read_text().splitlines() if line]
    reference_ids = {str(row["id"]) for row in records}
    baseline = predictions(args.baseline_predictions)
    candidate = predictions(args.candidate_predictions)
    if set(baseline) != reference_ids or set(candidate) != reference_ids:
        raise ValueError("Both prediction files must exactly cover the reference IDs")

    baseline_eligible = eligible_ids(records, baseline)
    candidate_eligible = eligible_ids(records, candidate)
    selected_ids = {
        str(row["id"])
        for row in records
        if str(row["id"]) in candidate_eligible - baseline_eligible
        and 1 in row.get("tag", [])
    }
    frames: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        frames[(str(row["scene_id"]), str(row["frame_id"]))].append(row)
    anchor_by_frame = {
        key: min(rows, key=lambda row: int(row["qa_index"])) for key, rows in frames.items()
    }

    output_rows: list[dict[str, Any]] = []
    tuple_counts: list[int] = []
    for record in records:
        if str(record["id"]) not in selected_ids:
            continue
        anchor = anchor_by_frame[(str(record["scene_id"]), str(record["frame_id"]))]
        tuples = TUPLE_RE.findall(text(anchor, "assistant"))
        if not tuples:
            raise ValueError(f"Oracle anchor has no complete tuple: {anchor['id']}")
        updated = copy.deepcopy(record)
        user_message = next(item for item in updated["messages"] if item["role"] == "user")
        user_message["content"] = (
            "DEV-ONLY ORACLE OBJECT GRAPH DIAGNOSTIC. Use the object graph as auxiliary "
            "frame context and answer the current question directly.\n"
            '<GRAPH_MEMORY source="oracle_reference_do_not_train">\n'
            + "\n".join(tuples)
            + "\n</GRAPH_MEMORY>\n<CURRENT_QUESTION>\n"
            + str(user_message["content"])
            + "\n</CURRENT_QUESTION>"
        )
        updated["v038b_probe"] = {
            "dev_only": True,
            "oracle_reference_used": True,
            "anchor_id": str(anchor["id"]),
            "tuple_count": len(tuples),
        }
        output_rows.append(updated)
        tuple_counts.append(len(tuples))

    output = Path(args.output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    baseline_output = Path(args.baseline_output_json)
    baseline_output.parent.mkdir(parents=True, exist_ok=True)
    baseline_output.write_text(json.dumps([
        {"id": str(row["id"]), "answer": baseline[str(row["id"])]}
        for row in output_rows
    ], ensure_ascii=False, indent=2) + "\n")
    judge_references = Path(args.judge_references_jsonl)
    judge_references.parent.mkdir(parents=True, exist_ok=True)
    selected_original = [row for row in records if str(row["id"]) in selected_ids]
    with judge_references.open("w", encoding="utf-8") as handle:
        for row in selected_original:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "schema_version": "drivelm-v038b-oracle-memory-probe-v1",
        "status": "dev_only_not_for_training_or_promotion",
        "reference_count": len(records),
        "baseline_eligible": len(baseline_eligible),
        "candidate_eligible": len(candidate_eligible),
        "newly_eligible": len(candidate_eligible - baseline_eligible),
        "selected_newly_eligible_planning": len(output_rows),
        "memory_tuple_min": min(tuple_counts),
        "memory_tuple_max": max(tuple_counts),
        "references_sha256": sha256(reference_path),
        "output_sha256": sha256(output),
        "baseline_output_sha256": sha256(baseline_output),
        "judge_references_sha256": sha256(judge_references),
    }
    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if len(output_rows) != 35:
        raise ValueError(f"Frozen v0.38B probe expected 35 rows, got {len(output_rows)}")


if __name__ == "__main__":
    main()
