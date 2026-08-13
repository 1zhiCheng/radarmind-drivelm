#!/usr/bin/env python3
"""Select one v0.38A checkpoint using frozen offline/structural gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


STEPS = (25, 50, 75, 100)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-offline", required=True)
    parser.add_argument("--baseline-structural", required=True)
    parser.add_argument("--sweep-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--max-token-f1-regression", type=float, default=0.002)
    parser.add_argument("--max-planning-token-f1-regression", type=float, default=0.002)
    return parser.parse_args()


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def offline_metrics(row: dict) -> dict[str, float]:
    return {
        "coverage": row["coverage"],
        "exact_match": row["overall"]["exact_match"],
        "token_f1": row["overall"]["token_f1"],
        "rouge_l": row["overall"]["rouge_l"],
        "planning_token_f1": row["by_task"]["planning"]["token_f1"],
        "mc_accuracy": row["multiple_choice"]["accuracy"],
    }


def structural_metrics(row: dict) -> dict[str, float]:
    return {
        "eligible_count": row["eligible_count"],
        "anchor_coordinate_f1": row["anchor_coordinate_macro"]["f1"],
        "tag3_coordinate_f1": row["tag3_coordinate_macro"]["f1"],
    }


def main() -> None:
    args = parse_args()
    baseline = {
        **offline_metrics(load(args.baseline_offline)),
        **structural_metrics(load(args.baseline_structural)),
    }
    sweep = Path(args.sweep_dir)
    candidates = []
    for step in STEPS:
        current = {
            **offline_metrics(load(sweep / f"checkpoint-{step}-dev-metrics.json")),
            **structural_metrics(load(sweep / f"checkpoint-{step}-structural.json")),
        }
        delta = {key: current[key] - baseline[key] for key in current}
        gates = {
            "coverage_100_percent": current["coverage"] == 1.0,
            "eligible_not_below_v037b": delta["eligible_count"] >= 0,
            "tag3_coordinate_f1_strictly_improved": delta["tag3_coordinate_f1"] > 0,
            "mc_not_below_v037b": delta["mc_accuracy"] >= 0,
            "token_f1_within_guardrail": (
                delta["token_f1"] >= -args.max_token_f1_regression
            ),
            "planning_token_f1_within_guardrail": (
                delta["planning_token_f1"]
                >= -args.max_planning_token_f1_regression
            ),
        }
        candidates.append({
            "step": step,
            "metrics": current,
            "delta": delta,
            "pre_judge_gates": gates,
            "eligible_for_judge": all(gates.values()),
        })

    eligible = [row for row in candidates if row["eligible_for_judge"]]
    eligible.sort(
        key=lambda row: (
            row["metrics"]["tag3_coordinate_f1"],
            row["metrics"]["eligible_count"],
            row["metrics"]["anchor_coordinate_f1"],
            row["metrics"]["token_f1"],
        ),
        reverse=True,
    )
    selected = eligible[0]["step"] if eligible else None
    report = {
        "schema_version": "drivelm-v038a-offline-selection-v1",
        "baseline": baseline,
        "candidates": candidates,
        "selected_step": selected,
        "selection_policy": (
            "Require full coverage, eligible >= v0.37B, tag-3 coordinate F1 > v0.37B, "
            "MC >= v0.37B, and lexical guardrails; rank coordinate F1, eligible, "
            "anchor F1, then Token-F1. DeepSeek scores are not visible to this selector."
        ),
        "thresholds": {
            "max_token_f1_regression": args.max_token_f1_regression,
            "max_planning_token_f1_regression": args.max_planning_token_f1_regression,
        },
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print("NONE" if selected is None else selected)


if __name__ == "__main__":
    main()
