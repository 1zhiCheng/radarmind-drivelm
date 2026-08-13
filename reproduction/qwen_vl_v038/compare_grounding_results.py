#!/usr/bin/env python3
"""Apply frozen v0.38A full-protocol promotion gates against v0.37B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-offline", required=True)
    parser.add_argument("--candidate-offline", required=True)
    parser.add_argument("--baseline-structural", required=True)
    parser.add_argument("--candidate-structural", required=True)
    parser.add_argument("--baseline-drivelm-ds", required=True)
    parser.add_argument("--candidate-drivelm-ds", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    return parser.parse_args()


def load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def extract(offline: dict, structural: dict, ds: dict) -> dict:
    return {
        "coverage": offline["coverage"],
        "exact_match": offline["overall"]["exact_match"],
        "token_f1": offline["overall"]["token_f1"],
        "rouge_l": offline["overall"]["rouge_l"],
        "mc_accuracy": offline["multiple_choice"]["accuracy"],
        "eligible_count": structural["eligible_count"],
        "anchor_coordinate_f1": structural["anchor_coordinate_macro"]["f1"],
        "tag3_coordinate_f1": structural["tag3_coordinate_macro"]["f1"],
        "planning": ds["metrics"]["planning_deepseek_100"],
        "graph": ds["metrics"]["graph_deepseek_100"],
        "match": ds["metrics"]["match_score_100"],
        "final": ds["metrics"]["drivelm_ds_final"],
        "judge_complete": bool(ds["judge"]["complete"]),
    }


def main() -> None:
    args = parse_args()
    baseline = extract(
        load(args.baseline_offline), load(args.baseline_structural),
        load(args.baseline_drivelm_ds),
    )
    candidate = extract(
        load(args.candidate_offline), load(args.candidate_structural),
        load(args.candidate_drivelm_ds),
    )
    numeric = tuple(key for key in baseline if key != "judge_complete")
    delta = {key: candidate[key] - baseline[key] for key in numeric}
    gates = {
        "coverage_100_percent": candidate["coverage"] == 1.0,
        "judge_complete": candidate["judge_complete"],
        "final_strictly_improved": delta["final"] > 0.0,
        "eligible_not_below_v037b": delta["eligible_count"] >= 0,
        "tag3_coordinate_f1_strictly_improved": delta["tag3_coordinate_f1"] > 0.0,
        "planning_regression_at_most_0_5_points": delta["planning"] >= -0.5,
        "mc_not_below_v037b": delta["mc_accuracy"] >= 0.0,
    }
    report = {
        "schema_version": "drivelm-v038a-comparison-v1",
        "baseline": baseline,
        "candidate": candidate,
        "delta": delta,
        "promotion_gates": gates,
        "promoted": all(gates.values()),
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    rows = []
    for key, label, scale in (
        ("coverage", "Coverage", 100), ("exact_match", "Exact Match", 100),
        ("token_f1", "Token-F1", 100), ("rouge_l", "ROUGE-L", 100),
        ("mc_accuracy", "MC accuracy", 100), ("eligible_count", "Eligible QA", 1),
        ("anchor_coordinate_f1", "Anchor coordinate F1", 100),
        ("tag3_coordinate_f1", "Tag-3 coordinate F1", 100),
        ("planning", "Planning /100", 1), ("graph", "Graph /100", 1),
        ("match", "Match /100", 1), ("final", "DriveLM-DS Final", 1),
    ):
        rows.append(
            f"| {label} | {baseline[key] * scale:.4f} | "
            f"{candidate[key] * scale:.4f} | {delta[key] * scale:+.4f} |"
        )
    markdown = "\n".join([
        "# v0.37B-75 vs v0.38A selected checkpoint", "",
        "| Metric | v0.37B | v0.38A | Delta |", "| --- | ---: | ---: | ---: |",
        *rows, "", f"Promotion: **{'PASS' if report['promoted'] else 'FAIL'}**", "",
        *[f"- [{'x' if value else ' '}] {key}" for key, value in gates.items()], "",
    ])
    Path(args.output_markdown).write_text(markdown)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
