#!/usr/bin/env python3
"""Dependency-free local metrics for a labeled DriveLM JSONL split.

This is a reproducible development evaluator, not the hidden challenge-server score.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from common import message_text, read_jsonl


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def token_f1(prediction: str, reference: str) -> float:
    pred = normalize(prediction).split()
    ref = normalize(reference).split()
    if not pred or not ref:
        return float(pred == ref)
    pred_counts = {x: pred.count(x) for x in set(pred)}
    ref_counts = {x: ref.count(x) for x in set(ref)}
    overlap = sum(min(pred_counts.get(x, 0), ref_counts.get(x, 0)) for x in pred_counts)
    if not overlap:
        return 0.0
    precision, recall = overlap / len(pred), overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def rouge_l(prediction: str, reference: str) -> float:
    pred, ref = normalize(prediction).split(), normalize(reference).split()
    if not pred or not ref:
        return float(pred == ref)
    row = [0] * (len(ref) + 1)
    for token in pred:
        previous = row[:]
        for j, target in enumerate(ref, start=1):
            row[j] = previous[j - 1] + 1 if token == target else max(previous[j], row[j - 1])
    lcs = row[-1]
    precision, recall = lcs / len(pred), lcs / len(ref)
    return 2 * precision * recall / (precision + recall) if lcs else 0.0


def summarize(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    if not rows:
        return {"count": 0}
    return {
        "count": len(rows),
        "exact_match": sum(x["exact"] for x in rows) / len(rows),
        "token_f1": sum(x["token_f1"] for x in rows) / len(rows),
        "rouge_l": sum(x["rouge_l"] for x in rows) / len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references-jsonl", required=True)
    parser.add_argument("--predictions-json", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    references = read_jsonl(args.references_jsonl)
    predictions = {x["id"]: str(x["answer"]) for x in json.loads(Path(args.predictions_json).read_text())}
    rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in references:
        if record["id"] not in predictions:
            continue
        pred, ref = predictions[record["id"]], message_text(record, "assistant")
        row = {
            "exact": float(normalize(pred) == normalize(ref)),
            "token_f1": token_f1(pred, ref),
            "rouge_l": rouge_l(pred, ref),
        }
        rows.append(row)
        grouped[record["task"]].append(row)
    multiple_choice = [r for r in references if normalize(message_text(r, "assistant")) in {"a", "b", "c", "d"}]
    mc_correct = sum(
        normalize(predictions.get(r["id"], "")) == normalize(message_text(r, "assistant"))
        for r in multiple_choice if r["id"] in predictions
    )
    report = {
        "note": "Offline labeled-dev metrics; not the official hidden-server score.",
        "reference_count": len(references),
        "prediction_count": len(predictions),
        "matched_ids": len(rows),
        "coverage": len(rows) / len(references),
        "overall": summarize(rows),
        "by_task": {task: summarize(values) for task, values in sorted(grouped.items())},
        "multiple_choice": {
            "matched": sum(r["id"] in predictions for r in multiple_choice),
            "accuracy": mc_correct / max(sum(r["id"] in predictions for r in multiple_choice), 1),
        },
    }
    Path(args.output_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
