#!/usr/bin/env python3
"""Build leakage-free planning trajectories from frozen upstream MoL predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from datasets import Dataset, Image, Sequence


CAMERAS = (
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
)
UPSTREAM_TASKS = {"perception", "prediction"}


def read_jsonl(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def message(row: dict[str, Any], role: str) -> str:
    return next(str(item["content"]) for item in row["messages"] if item["role"] == role)


def load_predictions(paths: list[str]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for path in paths:
        for row in json.loads(Path(path).read_text()):
            key = str(row["id"])
            if key in merged:
                raise ValueError(f"duplicate prediction id: {key}")
            merged[key] = str(row["answer"])
    return merged


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_split(
    source: str, prediction_paths: list[str], split: str, output: Path
) -> dict[str, Any]:
    records = read_jsonl(source)
    predictions = load_predictions(prediction_paths)
    frames: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        frames[(str(row["scene_id"]), str(row["frame_id"]))].append(row)

    required = {
        str(row["id"])
        for row in records
        if str(row["task"]) in UPSTREAM_TASKS
    }
    missing = required - set(predictions)
    extra = set(predictions) - required
    if missing or extra:
        raise ValueError(
            f"{split} upstream coverage mismatch: missing={len(missing)} extra={len(extra)}"
        )

    examples = []
    trajectory_lengths: dict[int, int] = defaultdict(int)
    for index, row in enumerate(records):
        if str(row["task"]) != "planning":
            continue
        ordered = sorted(
            frames[(str(row["scene_id"]), str(row["frame_id"]))],
            key=lambda item: int(item["qa_index"]),
        )
        upstream = [
            item for item in ordered
            if int(item["qa_index"]) < int(row["qa_index"])
            and str(item["task"]) in UPSTREAM_TASKS
        ]
        if not upstream:
            raise ValueError(f"planning row has no prior trajectory state: {row['id']}")
        state_lines = []
        for step in upstream:
            state_lines.append(
                f"Step {step['qa_index']} [{str(step['task']).upper()}]\n"
                f"Question: {message(step, 'user')}\n"
                f"Frozen model answer: {predictions[str(step['id'])]}"
            )
        trajectory_lengths[len(upstream)] += 1
        prompt = (
            "<image><image><image><image><image><image>\n"
            "The six images are synchronized surround views in this order: front, "
            "front-left, front-right, back, back-left, back-right.\n\n"
            "<TRAJECTORY_STATE source=\"frozen_v039b_mol_predictions\">\n"
            + "\n\n".join(state_lines)
            + "\n</TRAJECTORY_STATE>\n\n"
            "Answer the CURRENT PLANNING question using the images and trajectory state. "
            "Be concise, safety-aware, and do not invent objects.\n"
            f"Current question: {message(row, 'user')}"
        )
        image_paths = [str(row["images"][camera]) for camera in CAMERAS]
        if not all(Path(path).is_file() for path in image_paths):
            raise FileNotFoundError(f"missing image for {row['id']}")
        examples.append({
            "data_source": "radarmind_drivelm_trajectory_planning",
            "prompt": [
                {"role": "system", "content": message(row, "system")},
                {"role": "user", "content": prompt},
            ],
            "images": image_paths,
            "ability": "autonomous_driving_planning",
            "reward_model": {"style": "rule", "ground_truth": message(row, "assistant")},
            "extra_info": {
                "split": split,
                "index": len(examples),
                "id": str(row["id"]),
                "scene_id": str(row["scene_id"]),
                "frame_id": str(row["frame_id"]),
                "qa_index": int(row["qa_index"]),
                "upstream_steps": len(upstream),
                "question": message(row, "user"),
                "allowed_object_ids": sorted({
                    token[token.index("<"): token.index(">") + 1]
                    for text in [prompt]
                    for token in text.split()
                    if "<c" in token and ">" in token
                }),
            },
        })

    dataset = Dataset.from_list(examples).cast_column("images", Sequence(Image()))
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(str(output))
    scenes = {str(row["scene_id"]) for row in records}
    return {
        "split": split,
        "source": str(Path(source).resolve()),
        "source_sha256": sha256(source),
        "upstream_prediction_files": [str(Path(path).resolve()) for path in prediction_paths],
        "source_records": len(records),
        "source_scenes": len(scenes),
        "planning_trajectories": len(examples),
        "upstream_prediction_count": len(predictions),
        "upstream_coverage": 1.0,
        "trajectory_state_length_distribution": dict(sorted(trajectory_lengths.items())),
        "output": str(output.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--dev-jsonl", required=True)
    parser.add_argument("--train-predictions", nargs="+", required=True)
    parser.add_argument("--dev-predictions", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    train = build_split(args.train_jsonl, args.train_predictions, "train", output / "train.parquet")
    dev = build_split(args.dev_jsonl, args.dev_predictions, "dev", output / "dev.parquet")
    train_scenes = {str(row["scene_id"]) for row in read_jsonl(args.train_jsonl)}
    dev_scenes = {str(row["scene_id"]) for row in read_jsonl(args.dev_jsonl)}
    report = {
        "schema_version": "drivelm-v040-trajectory-dataset-v1",
        "state_source": "frozen v0.39B checkpoint-700 Perception/Prediction outputs",
        "gold_upstream_context_used": False,
        "dev_answers_used_for_training": False,
        "scene_overlap": len(train_scenes & dev_scenes),
        "train": train,
        "dev": dev,
    }
    if report["scene_overlap"]:
        raise ValueError("train/dev scene overlap")
    (output / "dataset_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
