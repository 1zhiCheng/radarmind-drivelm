#!/usr/bin/env python3
"""Evaluate v0.40 predictions with the exact training reward decomposition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Image, Sequence, load_dataset

from trajectory_reward import compute_score


METRICS = (
    "score", "token_f1_reward", "rouge_l_reward", "action_f1_reward",
    "exact_reward", "grounding_reward", "format_reward",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-parquet", required=True)
    parser.add_argument("--predictions-json", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    dataset = load_dataset("parquet", data_files=args.dev_parquet, split="train")
    dataset = dataset.cast_column("images", Sequence(Image(decode=False)))
    predictions = {
        str(row["id"]): str(row["answer"])
        for row in json.loads(Path(args.predictions_json).read_text())
    }
    sums = {metric: 0.0 for metric in METRICS}
    matched = 0
    for row in dataset:
        record_id = str(row["extra_info"]["id"])
        if record_id not in predictions:
            continue
        metrics = compute_score(
            data_source=str(row["data_source"]),
            solution_str=predictions[record_id],
            ground_truth=str(row["reward_model"]["ground_truth"]),
            extra_info=row["extra_info"],
        )
        matched += 1
        for metric in METRICS:
            sums[metric] += float(metrics[metric])
    report = {
        "schema_version": "radarmind-v040-trajectory-eval-v1",
        "reference_count": len(dataset),
        "prediction_count": len(predictions),
        "matched_ids": matched,
        "coverage": matched / len(dataset),
        "metrics": {metric: sums[metric] / max(matched, 1) for metric in METRICS},
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
