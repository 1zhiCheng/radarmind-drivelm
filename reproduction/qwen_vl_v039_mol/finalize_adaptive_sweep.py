#!/usr/bin/env python3
"""Freeze the automatically selected v0.39B policy or fall back to v0.39A."""

from __future__ import annotations
import argparse, json
from pathlib import Path


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--monitor", required=True)
    p.add_argument("--full-baseline", required=True)
    p.add_argument("--full-candidate")
    p.add_argument("--paired-comparison")
    p.add_argument("--model-root", required=True)
    p.add_argument("--v039a-root", required=True)
    p.add_argument("--output-json", required=True)
    a = p.parse_args()
    monitor = load(a.monitor)
    step = monitor["best_step"]
    promoted = False
    validation = {"needed": step is not None}
    if step is not None:
        full_base, full_candidate = load(a.full_baseline), load(a.full_candidate)
        paired = load(a.paired_comparison)
        full_delta = (
            full_candidate["metrics"]["drivelm_ds_final"]
            - full_base["metrics"]["drivelm_ds_final"]
        )
        validation.update({
            "full_drivelm_ds_delta": full_delta,
            "full_candidate_judge_complete": full_candidate["judge"]["complete"],
            "paired_audit_passed": paired["paired_audit_passed"],
        })
        promoted = bool(full_delta > 0 and full_candidate["judge"]["complete"] and paired["paired_audit_passed"])
    selected_step = step if promoted else None
    root = Path(a.model_root if promoted else a.v039a_root)
    paths = {
        task: str(root / f"expert_{task}" / (f"checkpoint-{selected_step}" if promoted else ""))
        for task in ("perception", "prediction", "planning", "behavior")
    }
    report = {
        "schema_version": "drivelm-v039b-adaptive-best-checkpoint-v1",
        "selected_source": f"v039b_checkpoint-{step}" if promoted else "v039a_mol",
        "selected_step": selected_step,
        "extended_candidate_promoted": promoted,
        "expert_adapter_paths": paths,
        "monitor_decision": monitor["decision"],
        "monitor_best_source": monitor["best_source"],
        "validation": validation,
        "next_stage": "trajectory RL GRPO/GSPO initialization",
    }
    Path(a.output_json).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
