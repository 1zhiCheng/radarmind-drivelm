#!/usr/bin/env python3
"""Extract every DriveLM JSONL record for one scene token."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--scene-token", required=True)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected: list[dict] = []
    with args.input_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("scene_id") == args.scene_token:
                selected.append(record)
    if not selected:
        raise ValueError(f"Scene {args.scene_token} is not present in {args.input_jsonl}")

    frame_count = len({row["frame_id"] for row in selected})
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for record in selected:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({
        "scene_token": args.scene_token,
        "records": len(selected),
        "frames": frame_count,
        "output": str(args.output_jsonl),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
