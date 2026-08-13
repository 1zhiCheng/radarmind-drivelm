#!/usr/bin/env python3
"""Compare B10 and B10-DPO and apply the v0.37A promotion gate."""

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
    parser.add_argument("--max-planning-regression", type=float, default=1.0)
    parser.add_argument("--max-mc-regression", type=float, default=0.01)
    return parser.parse_args()


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    base_offline = load(args.baseline_offline)
    cand_offline = load(args.candidate_offline)
    base_ds = load(args.baseline_drivelm_ds)
    cand_ds = load(args.candidate_drivelm_ds)
    base = {
        "coverage": base_offline["coverage"],
        "exact_match": base_offline["overall"]["exact_match"],
        "token_f1": base_offline["overall"]["token_f1"],
        "rouge_l": base_offline["overall"]["rouge_l"],
        "mc_accuracy": base_offline["multiple_choice"]["accuracy"],
        "planning": base_ds["metrics"]["planning_deepseek_100"],
        "final": base_ds["metrics"]["drivelm_ds_final"],
        "judge_complete": bool(base_ds["judge"]["complete"]),
    }
    candidate = {
        "coverage": cand_offline["coverage"],
        "exact_match": cand_offline["overall"]["exact_match"],
        "token_f1": cand_offline["overall"]["token_f1"],
        "rouge_l": cand_offline["overall"]["rouge_l"],
        "mc_accuracy": cand_offline["multiple_choice"]["accuracy"],
        "planning": cand_ds["metrics"]["planning_deepseek_100"],
        "final": cand_ds["metrics"]["drivelm_ds_final"],
        "judge_complete": bool(cand_ds["judge"]["complete"]),
    }
    deltas = {
        key: candidate[key] - base[key]
        for key in (
            "coverage", "exact_match", "token_f1", "rouge_l",
            "mc_accuracy", "planning", "final",
        )
    }
    gates = {
        "coverage_100_percent": candidate["coverage"] == 1.0,
        "judge_complete": candidate["judge_complete"],
        "final_improved": deltas["final"] > 0,
        "planning_non_material_regression": (
            deltas["planning"] >= -args.max_planning_regression
        ),
        "mc_non_material_regression": (
            deltas["mc_accuracy"] >= -args.max_mc_regression
        ),
    }
    promoted = all(gates.values())
    report = {
        "schema_version": "drivelm-v037a-comparison-v1",
        "baseline": base,
        "candidate": candidate,
        "delta": deltas,
        "promotion_gates": gates,
        "promoted": promoted,
        "thresholds": {
            "max_planning_regression_points": args.max_planning_regression,
            "max_mc_regression": args.max_mc_regression,
        },
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    rows = []
    for key, label, scale in (
        ("coverage", "Coverage", 100),
        ("exact_match", "Exact Match", 100),
        ("token_f1", "Token-F1", 100),
        ("rouge_l", "ROUGE-L", 100),
        ("mc_accuracy", "MC accuracy", 100),
        ("planning", "Planning /100", 1),
        ("final", "DriveLM-DS Final", 1),
    ):
        rows.append(
            f"| {label} | {base[key] * scale:.4f} | "
            f"{candidate[key] * scale:.4f} | {deltas[key] * scale:+.4f} |"
        )
    markdown = "\n".join(
        [
            "# B10 vs B10-DPO",
            "",
            "| Metric | B10 | B10-DPO | Delta |",
            "| --- | ---: | ---: | ---: |",
            *rows,
            "",
            f"Promotion decision: **{'PASS' if promoted else 'FAIL'}**",
            "",
            "## Gates",
            "",
            *[f"- [{'x' if value else ' '}] {key}" for key, value in gates.items()],
            "",
        ]
    )
    Path(args.output_markdown).write_text(markdown, encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
