#!/usr/bin/env python3
"""Summarize a same-ID graph-gating audit and apply paired promotion gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-report", required=True)
    parser.add_argument("--baseline-offline", required=True)
    parser.add_argument("--candidate-offline", required=True)
    parser.add_argument("--baseline-drivelm-ds", required=True)
    parser.add_argument("--candidate-drivelm-ds", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def main() -> None:
    args = parse_args()
    subset = load(args.subset_report)
    bo, co = load(args.baseline_offline), load(args.candidate_offline)
    bd, cd = load(args.baseline_drivelm_ds), load(args.candidate_drivelm_ds)
    baseline = {
        "exact_match": bo["overall"]["exact_match"],
        "token_f1": bo["overall"]["token_f1"],
        "rouge_l": bo["overall"]["rouge_l"],
        "mc_accuracy": bo["multiple_choice"]["accuracy"],
        "accuracy": bd["metrics"]["accuracy"],
        "planning": bd["metrics"]["planning_deepseek_100"],
        "language": bd["metrics"]["language_combined"],
        "coordinate_f1": bd["metrics"]["coordinate_f1_macro"],
        "graph": bd["metrics"]["graph_deepseek_100"],
        "match": bd["metrics"]["match_score_100"],
        "final": bd["metrics"]["drivelm_ds_final"],
    }
    candidate = {
        "exact_match": co["overall"]["exact_match"],
        "token_f1": co["overall"]["token_f1"],
        "rouge_l": co["overall"]["rouge_l"],
        "mc_accuracy": co["multiple_choice"]["accuracy"],
        "accuracy": cd["metrics"]["accuracy"],
        "planning": cd["metrics"]["planning_deepseek_100"],
        "language": cd["metrics"]["language_combined"],
        "coordinate_f1": cd["metrics"]["coordinate_f1_macro"],
        "graph": cd["metrics"]["graph_deepseek_100"],
        "match": cd["metrics"]["match_score_100"],
        "final": cd["metrics"]["drivelm_ds_final"],
    }
    delta = {key: candidate[key] - baseline[key] for key in baseline}
    gates = {
        "same_reference_and_prediction_count": (
            bo["reference_count"] == bo["prediction_count"]
            == co["reference_count"] == co["prediction_count"]
            == subset["common_eligible"]
        ),
        "both_judges_complete": bool(bd["judge"]["complete"] and cd["judge"]["complete"]),
        "final_strictly_improved": delta["final"] > 0.0,
        "planning_regression_at_most_0_5_points": delta["planning"] >= -0.5,
        "coordinate_f1_regression_at_most_0_5pp": delta["coordinate_f1"] >= -0.005,
        "mc_not_below_b10": delta["mc_accuracy"] >= 0.0,
    }
    report = {
        "schema_version": "drivelm-v037b-common-gating-comparison-v1",
        "subset": subset,
        "baseline": baseline,
        "candidate": candidate,
        "delta": delta,
        "judge_completed": {
            "baseline": bd["judge"]["completed"],
            "candidate": cd["judge"]["completed"],
        },
        "paired_gates": gates,
        "paired_audit_passed": all(gates.values()),
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
