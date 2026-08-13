#!/usr/bin/env python3
"""Create deterministic dev shards or merge prediction shards in source order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    shard = subparsers.add_parser("shard")
    shard.add_argument("--input-jsonl", required=True)
    shard.add_argument("--output-dir", required=True)
    shard.add_argument("--num-shards", type=int, default=3)
    merge = subparsers.add_parser("merge")
    merge.add_argument("--source-jsonl", required=True)
    merge.add_argument("--prediction-json", action="append", required=True)
    merge.add_argument("--output-json", required=True)
    return parser.parse_args()


def shard(args: argparse.Namespace) -> None:
    if args.num_shards <= 0:
        raise ValueError("num shards must be positive")
    rows = read_jsonl(args.input_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    handles = [
        (output_dir / f"dev_shard{index}.jsonl").open("w", encoding="utf-8")
        for index in range(args.num_shards)
    ]
    try:
        for index, row in enumerate(rows):
            handles[index % args.num_shards].write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )
    finally:
        for handle in handles:
            handle.close()
    print(json.dumps({"records": len(rows), "num_shards": args.num_shards}))


def merge(args: argparse.Namespace) -> None:
    source = read_jsonl(args.source_jsonl)
    source_ids = [str(row["id"]) for row in source]
    by_id: dict[str, dict] = {}
    for path in args.prediction_json:
        predictions = json.loads(Path(path).read_text(encoding="utf-8"))
        for prediction in predictions:
            prediction_id = str(prediction["id"])
            if prediction_id in by_id:
                raise ValueError(f"Duplicate prediction id {prediction_id}")
            by_id[prediction_id] = prediction
    missing = set(source_ids) - set(by_id)
    unexpected = set(by_id) - set(source_ids)
    if missing or unexpected or len(source_ids) != len(set(source_ids)):
        raise ValueError(
            f"Prediction coverage error: missing={len(missing)}, "
            f"unexpected={len(unexpected)}"
        )
    ordered = [by_id[row_id] for row_id in source_ids]
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "source_records": len(source_ids),
                "predictions": len(ordered),
                "coverage": len(ordered) / len(source_ids),
                "output": str(output),
            }
        )
    )


def main() -> None:
    args = parse_args()
    if args.command == "shard":
        shard(args)
    else:
        merge(args)


if __name__ == "__main__":
    main()
