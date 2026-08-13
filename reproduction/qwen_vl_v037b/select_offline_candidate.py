#!/usr/bin/env python3
"""Summarize the v0.37B sweep and select one candidate for semantic judging."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--sweep-dir", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def metrics(row: dict) -> dict[str, float]:
    return {
        "coverage": row["coverage"],
        "exact_match": row["overall"]["exact_match"],
        "token_f1": row["overall"]["token_f1"],
        "rouge_l": row["overall"]["rouge_l"],
        "planning_token_f1": row["by_task"]["planning"]["token_f1"],
        "mc_accuracy": row["multiple_choice"]["accuracy"],
    }


def main() -> None:
    args = parse_args()
    baseline = metrics(json.loads(Path(args.baseline).read_text()))
    sweep = Path(args.sweep_dir)
    candidates = []
    for step in (25, 50, 75, 100):
        path = sweep / f"checkpoint-{step}-dev-metrics.json"
        current = metrics(json.loads(path.read_text()))
        delta = {key: current[key] - baseline[key] for key in current}
        pre_judge_gates = {
            "coverage_100_percent": current["coverage"] == 1.0,
            "token_f1_not_below_b10": delta["token_f1"] >= 0.0,
            "planning_token_f1_not_below_b10": delta["planning_token_f1"] >= 0.0,
            "mc_improved_over_b10": delta["mc_accuracy"] > 0.0,
        }
        candidates.append({
            "step": step,
            "metrics": current,
            "delta": delta,
            "pre_judge_gates": pre_judge_gates,
            "eligible_for_judge": all(pre_judge_gates.values()),
        })

    eligible = [row for row in candidates if row["eligible_for_judge"]]
    # Favor the main lexical metric, then planning stability, then MC accuracy.
    eligible.sort(
        key=lambda row: (
            row["metrics"]["token_f1"],
            row["metrics"]["planning_token_f1"],
            row["metrics"]["mc_accuracy"],
        ),
        reverse=True,
    )
    selected = eligible[0]["step"] if eligible else None
    report = {
        "schema_version": "drivelm-v037b-offline-selection-v1",
        "baseline": baseline,
        "candidates": candidates,
        "selected_step": selected,
        "selection_policy": (
            "Require full coverage, overall Token-F1 >= B10, planning Token-F1 "
            ">= B10, and MC accuracy > B10; rank by Token-F1, planning Token-F1, MC."
        ),
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print("NONE" if selected is None else selected)


if __name__ == "__main__":
    main()
