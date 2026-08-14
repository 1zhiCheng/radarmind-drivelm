#!/usr/bin/env python3
"""Select a MoL checkpoint and decide whether the sweep should be extended."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def flatten(offline: dict, structural: dict) -> dict[str, float]:
    return {
        "coverage": offline["coverage"],
        "token_f1": offline["overall"]["token_f1"],
        "rouge_l": offline["overall"]["rouge_l"],
        "planning_token_f1": offline["by_task"]["planning"]["token_f1"],
        "mc_accuracy": offline["multiple_choice"]["accuracy"],
        "eligible_count": structural["eligible_count"],
        "anchor_coordinate_f1": structural["anchor_coordinate_macro"]["f1"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", required=True)
    parser.add_argument("--baseline-offline", required=True)
    parser.add_argument("--baseline-structural", required=True)
    parser.add_argument("--steps", nargs="+", type=int, required=True)
    parser.add_argument("--min-delta", type=float, default=0.0005)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--max-step-cap", type=int, default=1500)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    directory = Path(args.evaluation_dir)
    baseline = flatten(load(Path(args.baseline_offline)), load(Path(args.baseline_structural)))
    points = []
    # The already promoted v0.39A model is a real incumbent. A longer run must
    # beat it by min_delta; otherwise the controller keeps v0.39A.
    best_score = baseline["token_f1"]
    best_step = None
    stale = 0
    for step in sorted(args.steps):
        metrics = flatten(
            load(directory / f"checkpoint-{step}_offline.json"),
            load(directory / f"checkpoint-{step}_structural.json"),
        )
        delta = {key: metrics[key] - baseline[key] for key in metrics}
        gates = {
            "coverage_100_percent": metrics["coverage"] == 1.0,
            "planning_regression_at_most_0_2pp": delta["planning_token_f1"] >= -0.002,
            "mc_regression_at_most_0_2pp": delta["mc_accuracy"] >= -0.002,
            "eligible_regression_at_most_20": delta["eligible_count"] >= -20,
            "anchor_f1_regression_at_most_0_2pp": delta["anchor_coordinate_f1"] >= -0.002,
        }
        hard_safety_failure = (
            metrics["coverage"] != 1.0
            or delta["planning_token_f1"] < -0.005
            or delta["mc_accuracy"] < -0.005
            or delta["eligible_count"] < -50
            or delta["anchor_coordinate_f1"] < -0.005
        )
        eligible = all(gates.values())
        improved = eligible and metrics["token_f1"] > best_score + args.min_delta
        if improved:
            best_score = metrics["token_f1"]
            best_step = step
            stale = 0
        else:
            stale += 1
        points.append({
            "step": step,
            "metrics": metrics,
            "delta_vs_v039a": delta,
            "safety_gates": gates,
            "eligible": eligible,
            "hard_safety_failure": hard_safety_failure,
            "significant_improvement": improved,
            "stale_evaluations": stale,
        })

    latest = max(args.steps)
    safety_stop = bool(points[-1]["hard_safety_failure"])
    stop = safety_stop or stale >= args.patience or latest >= args.max_step_cap
    decision = "stop_and_promote_best" if stop else "extend_sweep"
    report = {
        "schema_version": "drivelm-v039b-adaptive-mol-monitor-v1",
        "selection_metric": "overall token_f1 subject to planning/MC/graph-grounding safety gates",
        "min_delta": args.min_delta,
        "patience": args.patience,
        "hard_max_step_cap": args.max_step_cap,
        "baseline": baseline,
        "points": points,
        "best_step": best_step,
        "best_source": f"checkpoint-{best_step}" if best_step is not None else "v039a_mol",
        "best_token_f1": best_score,
        "decision": decision,
        "reason": (
            "latest checkpoint crossed a hard safety-regression threshold"
            if safety_stop
            else f"{stale} consecutive checkpoints without >= {args.min_delta} token-F1 gain"
            if stale >= args.patience
            else ("hard cap reached" if latest >= args.max_step_cap else "patience not exhausted")
        ),
        "judge_policy": "Run DeepSeek only for the selected point; promotion still requires full and same-ID DriveLM-DS.",
    }
    Path(args.output_json).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
