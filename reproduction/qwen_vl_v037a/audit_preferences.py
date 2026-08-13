#!/usr/bin/env python3
"""Enforce minimum quality and task coverage for v0.37A preferences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TASKS = ("perception", "prediction", "planning", "behavior")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--min-pairs", type=int, default=500)
    parser.add_argument("--min-pairs-per-task", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = json.loads(Path(args.report_json).read_text(encoding="utf-8"))
    checks = {
        "minimum_total_pairs": report["preference_pairs"] >= args.min_pairs,
        "zero_train_dev_scene_overlap": report["train_dev_scene_overlap"] == 0,
        "zero_candidate_ids_outside_train": report["candidate_ids_outside_train"] == 0,
        "zero_dev_ids_in_preferences": report["dev_ids_in_preferences"] == 0,
        "no_external_semantic_api": report["semantic_api_used"] is False,
    }
    for task in TASKS:
        checks[f"minimum_{task}_pairs"] = (
            int(report["by_task"].get(task, 0)) >= args.min_pairs_per_task
        )
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "preference_pairs": report["preference_pairs"],
        "by_task": report["by_task"],
        "thresholds": {
            "min_pairs": args.min_pairs,
            "min_pairs_per_task": args.min_pairs_per_task,
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
