#!/usr/bin/env python3
"""Compare B10 with one v0.37B checkpoint using the frozen promotion gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-offline", required=True)
    parser.add_argument("--candidate-offline", required=True)
    parser.add_argument("--baseline-drivelm-ds", required=True)
    parser.add_argument("--candidate-drivelm-ds", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    return parser.parse_args()


def load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def extract(offline: dict, ds: dict) -> dict:
    return {
        "coverage": offline["coverage"],
        "exact_match": offline["overall"]["exact_match"],
        "token_f1": offline["overall"]["token_f1"],
        "rouge_l": offline["overall"]["rouge_l"],
        "mc_accuracy": offline["multiple_choice"]["accuracy"],
        "planning": ds["metrics"]["planning_deepseek_100"],
        "coordinate_f1": ds["metrics"]["coordinate_f1_macro"],
        "final": ds["metrics"]["drivelm_ds_final"],
        "judge_complete": bool(ds["judge"]["complete"]),
    }


def main() -> None:
    args = parse_args()
    baseline = extract(load(args.baseline_offline), load(args.baseline_drivelm_ds))
    candidate = extract(load(args.candidate_offline), load(args.candidate_drivelm_ds))
    numeric = (
        "coverage", "exact_match", "token_f1", "rouge_l", "mc_accuracy",
        "planning", "coordinate_f1", "final",
    )
    delta = {key: candidate[key] - baseline[key] for key in numeric}
    gates = {
        "coverage_100_percent": candidate["coverage"] == 1.0,
        "judge_complete": candidate["judge_complete"],
        "final_strictly_improved": delta["final"] > 0.0,
        "planning_regression_at_most_0_5_points": delta["planning"] >= -0.5,
        "coordinate_f1_regression_at_most_0_5pp": delta["coordinate_f1"] >= -0.005,
        "mc_not_below_b10": delta["mc_accuracy"] >= 0.0,
    }
    report = {
        "schema_version": "drivelm-v037b-comparison-v1",
        "baseline": baseline,
        "candidate": candidate,
        "delta": delta,
        "promotion_gates": gates,
        "promoted": all(gates.values()),
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2) + "\n")

    rows = []
    for key, label, scale in (
        ("coverage", "Coverage", 100),
        ("exact_match", "Exact Match", 100),
        ("token_f1", "Token-F1", 100),
        ("rouge_l", "ROUGE-L", 100),
        ("mc_accuracy", "MC accuracy", 100),
        ("planning", "Planning /100", 1),
        ("coordinate_f1", "Coordinate F1", 100),
        ("final", "DriveLM-DS Final", 1),
    ):
        rows.append(
            f"| {label} | {baseline[key] * scale:.4f} | "
            f"{candidate[key] * scale:.4f} | {delta[key] * scale:+.4f} |"
        )
    markdown = "\n".join([
        "# B10 vs v0.37B selected checkpoint", "",
        "| Metric | B10 | Candidate | Delta |",
        "| --- | ---: | ---: | ---: |", *rows, "",
        f"Promotion decision: **{'PASS' if report['promoted'] else 'FAIL'}**", "",
        "## Frozen gates", "",
        *[f"- [{'x' if value else ' '}] {key}" for key, value in gates.items()], "",
    ])
    Path(args.output_markdown).write_text(markdown)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
