#!/usr/bin/env python3
"""Build a train-only hard-task curriculum for DriveLM Stage-2 SFT."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from analyze_errors import question_family
from common import read_jsonl


DEFAULT_WEIGHTS = {
    "important_objects": 1,
    "notice_graph": 2,
    "collision_reasoning": 2,
    "safe_actions": 1,
    "behavior_mc": 1,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--weights-json", help="JSON object mapping family names to integer repeats")
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()

    weights = DEFAULT_WEIGHTS if not args.weights_json else json.loads(args.weights_json)
    if not weights or any(not isinstance(value, int) or value < 0 for value in weights.values()):
        raise ValueError("Every curriculum weight must be a non-negative integer")
    source = read_jsonl(args.input_jsonl)
    source_counts = Counter(question_family(record) for record in source)
    selected = []
    selected_counts: Counter[str] = Counter()
    for record in source:
        family = question_family(record)
        repeat = weights.get(family, 0)
        selected.extend([record] * repeat)
        selected_counts[family] += repeat
    random.Random(args.seed).shuffle(selected)

    output = Path(args.output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in selected:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    report = {
        "source_jsonl": str(Path(args.input_jsonl).resolve()),
        "output_jsonl": str(output.resolve()),
        "seed": args.seed,
        "weights": weights,
        "source_records": len(source),
        "output_records": len(selected),
        "source_family_counts": dict(sorted(source_counts.items())),
        "output_family_counts": dict(sorted(selected_counts.items())),
        "excluded_families": sorted(set(source_counts) - set(weights)),
        "dev_records_used": 0,
    }
    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
