#!/usr/bin/env python3
"""Compare the frozen v0.40 baseline, GRPO-70 and GSPO-90 evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


NAMES = ("baseline", "grpo70", "gspo90")


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()
    results = {}
    for name in NAMES:
        trajectory = load(args.eval_dir / "metrics" / f"{name}_trajectory.json")
        offline = load(args.eval_dir / "metrics" / f"{name}_offline.json")
        semantic = load(args.eval_dir / "metrics" / f"{name}_drivelm_ds.json")
        results[name] = {
            "coverage": trajectory["coverage"],
            **trajectory["metrics"],
            "offline_exact": offline["overall"]["exact_match"],
            "offline_token_f1": offline["overall"]["token_f1"],
            "offline_rouge_l": offline["overall"]["rouge_l"],
            "planning_deepseek_100": semantic["metrics"]["planning_deepseek_100"],
            "judge_complete": bool(semantic["judge"]["complete"]),
            "judge_completed": int(semantic["judge"]["completed"]),
        }
    baseline = results["baseline"]
    candidates = {}
    for name in ("grpo70", "gspo90"):
        row = results[name]
        delta = {
            key: row[key] - baseline[key]
            for key in (
                "score", "token_f1_reward", "rouge_l_reward", "action_f1_reward",
                "exact_reward", "planning_deepseek_100",
            )
        }
        gates = {
            "coverage_100_percent": row["coverage"] == 1.0,
            "judge_complete": row["judge_complete"],
            "trajectory_reward_strictly_improved": delta["score"] > 0.0,
            "token_f1_strictly_improved": delta["token_f1_reward"] > 0.0,
            "planning_deepseek_regression_at_most_0_5": delta["planning_deepseek_100"] >= -0.5,
        }
        candidates[name] = {"delta_vs_baseline": delta, "promotion_gates": gates, "eligible": all(gates.values())}
    eligible = [name for name, value in candidates.items() if value["eligible"]]
    eligible.sort(
        key=lambda name: (results[name]["planning_deepseek_100"], results[name]["score"]),
        reverse=True,
    )
    report = {
        "schema_version": "radarmind-v040-final-comparison-v1",
        "scope": "1,399 scene-isolated planning trajectory dev rows; not the hidden challenge score",
        "frozen_policy": (
            "Require 100% coverage/judge completion, strict local reward and token-F1 gains, and no more "
            "than 0.5 DeepSeek planning points regression. Rank passing candidates by planning judge, "
            "then trajectory reward."
        ),
        "results": results,
        "candidates": candidates,
        "promoted": eligible[0] if eligible else None,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n")
    rows = []
    labels = (
        ("score", "Trajectory reward", 100),
        ("token_f1_reward", "Token-F1", 100),
        ("rouge_l_reward", "ROUGE-L", 100),
        ("action_f1_reward", "Action-F1", 100),
        ("exact_reward", "Exact Match", 100),
        ("planning_deepseek_100", "Planning judge /100", 1),
    )
    for key, label, scale in labels:
        values = [results[name][key] * scale for name in NAMES]
        rows.append(f"| {label} | {values[0]:.4f} | {values[1]:.4f} | {values[2]:.4f} |")
    markdown = "\n".join([
        "# v0.40 trajectory RL final comparison", "",
        "> Local 1,399-row scene-isolated planning trajectory dev. DeepSeek-V4-Flash replaces the public GPT judge; this is not a hidden-server score.", "",
        "| Metric | Baseline | GRPO-70 | GSPO-90 |", "| --- | ---: | ---: | ---: |", *rows, "",
        f"Promoted model: **{report['promoted'] or 'NONE'}**", "", "## Frozen promotion gates", "",
        *[
            f"### {name}\n\n" + "\n".join(f"- [{'x' if passed else ' '}] {gate}" for gate, passed in candidates[name]["promotion_gates"].items())
            for name in ("grpo70", "gspo90")
        ], "",
    ])
    args.output_markdown.write_text(markdown)
    print(json.dumps({"promoted": report["promoted"], "results": results}, indent=2))


if __name__ == "__main__":
    main()
