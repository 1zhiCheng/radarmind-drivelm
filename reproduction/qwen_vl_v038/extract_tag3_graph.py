#!/usr/bin/env python3
"""Extract all train-only DriveLM tag-3 graph-coordinate QA records."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    source, output = Path(args.input_jsonl), Path(args.output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    scenes: set[str] = set()
    by_index: Counter[str] = Counter()
    with source.open(encoding="utf-8") as reader, output.open("w", encoding="utf-8") as writer:
        for line in reader:
            if not line.strip():
                continue
            row = json.loads(line)
            if 3 not in row.get("tag", []):
                continue
            writer.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
            scenes.add(str(row["scene_id"]))
            by_index[str(row["qa_index"])] += 1
    report = {
        "schema_version": "drivelm-v038-tag3-extract-v1",
        "source": str(source.resolve()),
        "source_sha256": sha256(source),
        "records": count,
        "scenes": len(scenes),
        "by_qa_index": dict(sorted(by_index.items())),
        "filter": "tag contains 3",
        "output_sha256": sha256(output),
    }
    output.with_suffix(output.suffix + ".report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
