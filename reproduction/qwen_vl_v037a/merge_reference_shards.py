#!/usr/bin/env python3
"""Merge and audit v0.37A precomputed reference shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preference-jsonl", required=True)
    parser.add_argument("--reference-jsonl", action="append", required=True)
    parser.add_argument("--output-jsonl", required=True)
    return parser.parse_args()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_ids(path: str | Path) -> list[str]:
    ids: list[str] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                ids.append(str(json.loads(line)["id"]))
    return ids


def main() -> None:
    args = parse_args()
    expected_order = load_ids(args.preference_jsonl)
    if len(expected_order) != len(set(expected_order)):
        raise ValueError("Preference file contains duplicate IDs")
    expected = set(expected_order)
    by_id: dict[str, dict] = {}
    normalization: str | None = None
    for path in args.reference_jsonl:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                row_id = str(row["id"])
                if row_id in by_id:
                    raise ValueError(f"Duplicate id {row_id} in {path}:{line_number}")
                current = str(row["reference_logp_normalization"])
                if normalization is None:
                    normalization = current
                elif normalization != current:
                    raise ValueError("Reference shards use different normalizations")
                by_id[row_id] = row
    missing = expected - set(by_id)
    unexpected = set(by_id) - expected
    if missing or unexpected:
        raise ValueError(
            f"Reference coverage mismatch: missing={len(missing)}, "
            f"unexpected={len(unexpected)}"
        )
    output = Path(args.output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row_id in expected_order:
            handle.write(json.dumps(by_id[row_id], ensure_ascii=False) + "\n")
    report = {
        "schema_version": "drivelm-v037a-reference-merge-v1",
        "expected_preferences": len(expected_order),
        "merged_records": len(by_id),
        "missing": len(missing),
        "unexpected": len(unexpected),
        "coverage": len(by_id) / len(expected) if expected else 0.0,
        "normalization": normalization,
        "preference_sha256": sha256(args.preference_jsonl),
        "output_sha256": sha256(output),
        "input_shards": [
            {"path": str(Path(path).resolve()), "sha256": sha256(path)}
            for path in args.reference_jsonl
        ],
    }
    report_path = output.with_suffix(output.suffix + ".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
