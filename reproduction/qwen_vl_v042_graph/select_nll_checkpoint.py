#!/usr/bin/env python3
"""Select the lowest-NLL Graph-SFT checkpoint before expensive rollout evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    rows = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.reports]
    counts = {(row["trajectories"], row["qa_nodes"]) for row in rows}
    if len(counts) != 1:
        raise ValueError(f"checkpoint screens use different dev scopes: {counts}")
    ranked = sorted(rows, key=lambda row: (float(row["token_mean_nll"]), row["adapter"]))
    report = {
        "schema_version": "radarmind-drivelm-v042-graph-nll-selection-v1",
        "selection_metric": "minimum full teacher-forced graph dev token NLL",
        "selected_adapter": ranked[0]["adapter"],
        "selected_nll": ranked[0]["token_mean_nll"],
        "ranked": ranked,
        "promotion_status": "not evaluated; predicted-context 3355-QA rollout and DriveLM-DS required",
    }
    Path(args.output_json).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
