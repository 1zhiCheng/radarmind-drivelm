#!/usr/bin/env python3
"""Merge Graph rollout shards into original QA order with exact coverage checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from graph_data import read_graph_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-jsonl", required=True)
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--report-json", required=True)
    args = parser.parse_args()
    records = read_graph_jsonl(args.graph_jsonl)
    expected = [str(node["id"]) for record in records for node in record["nodes"]]
    predictions: dict[str, dict[str, str]] = {}
    shard_counts: dict[str, int] = {}
    for value in args.shards:
        path = Path(value)
        rows = json.loads(path.read_text(encoding="utf-8"))
        shard_counts[str(path)] = len(rows)
        for row in rows:
            node_id = str(row["id"])
            if node_id in predictions:
                raise ValueError(f"duplicate prediction {node_id}")
            predictions[node_id] = {"id": node_id, "answer": str(row["answer"])}
    missing = sorted(set(expected) - set(predictions))
    extra = sorted(set(predictions) - set(expected))
    if missing or extra:
        raise ValueError(f"coverage mismatch: missing={len(missing)}, extra={len(extra)}")
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps([predictions[node_id] for node_id in expected], ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    report = {
        "schema_version": "radarmind-drivelm-v042-predicted-context-rollout-v1",
        "context_policy": "all upstream answers are model predictions; no gold answers",
        "expected": len(expected),
        "predictions": len(predictions),
        "coverage": 1.0,
        "shards": shard_counts,
    }
    Path(args.report_json).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
