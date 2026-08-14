#!/usr/bin/env python3
"""Summarize zero-shot and full-system GRPO/GSPO DriveLM results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--baseline-ds", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metrics(offline: dict, ds: dict) -> dict:
    return {
        "coverage": offline["coverage"],
        "exact_match": offline["overall"]["exact_match"],
        "token_f1": offline["overall"]["token_f1"],
        "rouge_l": offline["overall"]["rouge_l"],
        "mc_accuracy": offline["multiple_choice"]["accuracy"],
        "eligible": ds["gating"]["eligible_count"],
        "planning_deepseek_100": ds["metrics"]["planning_deepseek_100"],
        "coordinate_f1": ds["metrics"]["coordinate_f1_macro"],
        "drivelm_ds_final": ds["metrics"]["drivelm_ds_final"],
        "judge_complete": ds["judge"]["complete"],
        "judge_completed": ds["judge"]["completed"],
    }


def main() -> None:
    args = parse_args()
    root = Path(args.result_root)
    baseline_ds = load(Path(args.baseline_ds))
    results = {}
    for name in ("raw", "grpo70", "gspo90"):
        results[name] = metrics(
            load(root / "metrics" / f"{name}_offline.json"),
            load(root / "metrics" / f"{name}_drivelm_ds.json"),
        )
    results["mol700"] = metrics(
        load(root / "metrics" / "mol700_offline.json"), baseline_ds
    )
    baseline = results["mol700"]
    for name in ("grpo70", "gspo90"):
        results[name]["delta_vs_mol700"] = {
            key: results[name][key] - baseline[key]
            for key in (
                "exact_match",
                "token_f1",
                "rouge_l",
                "mc_accuracy",
                "planning_deepseek_100",
                "coordinate_f1",
                "drivelm_ds_final",
            )
        }
        paired = load(root / "same_id" / name / "common_gating_comparison.json")
        results[name]["same_id_audit_passed"] = paired["paired_audit_passed"]
        results[name]["same_id_final_delta"] = paired["delta"]["final"]
    candidates = [
        name
        for name in ("grpo70", "gspo90")
        if results[name]["judge_complete"]
        and results[name]["coverage"] == 1.0
        and results[name]["drivelm_ds_final"] > baseline["drivelm_ds_final"]
        and results[name]["same_id_audit_passed"]
    ]
    promoted = max(candidates, key=lambda x: results[x]["drivelm_ds_final"]) if candidates else "mol700"
    report = {
        "schema_version": "radarmind-v040-full-system-summary-v1",
        "scope": "3,355 scene-isolated all-task dev; local DriveLM-DS proxy",
        "results": results,
        "promotion_policy": "100% coverage and judge completion, strict full Final gain, and all same-ID gates",
        "promoted": promoted,
    }
    Path(args.output_json).write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# v0.40 full-system completion",
        "",
        "| Model | EM | Token-F1 | Planning | Final | Judge | Decision |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name in ("raw", "mol700", "grpo70", "gspo90"):
        row = results[name]
        decision = "**promoted**" if name == promoted else "control/candidate"
        lines.append(
            f"| {name} | {100*row['exact_match']:.4f} | {100*row['token_f1']:.4f} | "
            f"{row['planning_deepseek_100']:.4f} | {row['drivelm_ds_final']:.6f} | "
            f"{row['judge_completed']} | {decision} |"
        )
    Path(args.output_markdown).write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
