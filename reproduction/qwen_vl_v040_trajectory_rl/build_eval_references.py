#!/usr/bin/env python3
"""Build the exact 1,399-row planning reference set used by v0.40 dev."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Image, Sequence, load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-parquet", required=True)
    parser.add_argument("--source-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    args = parser.parse_args()
    dataset = load_dataset("parquet", data_files=args.dev_parquet, split="train")
    dataset = dataset.cast_column("images", Sequence(Image(decode=False)))
    ordered_ids = [str(row["extra_info"]["id"]) for row in dataset]
    source = {
        str(row["id"]): row
        for line in Path(args.source_jsonl).read_text().splitlines()
        if line.strip() and (row := json.loads(line))
    }
    missing = [record_id for record_id in ordered_ids if record_id not in source]
    if missing:
        raise ValueError(f"missing {len(missing)} planning references; first={missing[0]}")
    records = [source[record_id] for record_id in ordered_ids]
    if any(str(row["task"]) != "planning" for row in records):
        raise ValueError("non-planning row entered v0.40 reference set")
    output = Path(args.output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
    print(json.dumps({"output": str(output), "rows": len(records), "unique_ids": len(set(ordered_ids))}))


if __name__ == "__main__":
    main()
