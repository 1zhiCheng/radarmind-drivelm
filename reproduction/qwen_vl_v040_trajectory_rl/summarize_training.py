#!/usr/bin/env python3
"""Summarize deterministic v0.40 dev curves and freeze checkpoint selection."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


METRICS = {
    "reward": "val-core/radarmind_drivelm_trajectory_planning/reward/mean@1",
    "token_f1": "val-aux/radarmind_drivelm_trajectory_planning/token_f1_reward/mean@1",
    "rouge_l": "val-aux/radarmind_drivelm_trajectory_planning/rouge_l_reward/mean@1",
    "action_f1": "val-aux/radarmind_drivelm_trajectory_planning/action_f1_reward/mean@1",
    "exact": "val-aux/radarmind_drivelm_trajectory_planning/exact_reward/mean@1",
    "grounding": "val-aux/radarmind_drivelm_trajectory_planning/grounding_reward/mean@1",
    "format": "val-aux/radarmind_drivelm_trajectory_planning/format_reward/mean@1",
}


def parse_curve(path: Path) -> list[dict[str, float | int]]:
    rows: dict[int, dict[str, float | int]] = {}
    text = path.read_text(errors="replace").replace("\r", "\n")
    for line in text.splitlines():
        step_match = re.search(r"step:(\d+)", line)
        if step_match is None or METRICS["reward"] not in line:
            continue
        row: dict[str, float | int] = {"step": int(step_match.group(1))}
        for short, full in METRICS.items():
            match = re.search(re.escape(full) + r":(?:np\.float64\()?([-+0-9.eE]+)", line)
            if match is None:
                raise ValueError(f"missing {short} at step {row['step']} in {path}")
            row[short] = float(match.group(1))
        rows[int(row["step"])] = row
    if not rows:
        raise ValueError(f"no validation curve found in {path}")
    return [rows[step] for step in sorted(rows)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grpo-log", type=Path, required=True)
    parser.add_argument("--gspo-log", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    curves = {"grpo": parse_curve(args.grpo_log), "gspo": parse_curve(args.gspo_log)}
    selected = {
        name: max(rows, key=lambda row: (float(row["reward"]), float(row["token_f1"])))
        for name, rows in curves.items()
    }
    report = {
        "schema_version": "radarmind-v040-frozen-selection-v1",
        "selection_policy": (
            "Select the highest full-dev deterministic trajectory reward; break ties by token_f1. "
            "DeepSeek scores are not visible to checkpoint selection."
        ),
        "final_step_aggregate_note": (
            "VERL ran all 117 final validation batches at global step 100 but did not emit the "
            "aggregate line; selection therefore uses the fully aggregated step 0..90 reports."
        ),
        "curves": curves,
        "selected": selected,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["selected"], indent=2))


if __name__ == "__main__":
    main()
