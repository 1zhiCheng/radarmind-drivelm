#!/usr/bin/env python3
"""Break DriveLM predictions down by question family and object-ID structure."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from common import message_text, read_jsonl
from evaluate_offline import normalize, rouge_l, token_f1


OBJECT_ID_RE = re.compile(r"<(c\d+),(CAM_[A-Z_]+),", flags=re.IGNORECASE)


def question_family(record: dict[str, Any]) -> str:
    question = message_text(record, "user").lower()
    if "please select the correct answer" in question:
        return "behavior_mc" if record["task"] == "behavior" else "moving_status_mc"
    if "what are the important objects" in question:
        return "important_objects"
    if "what object should the ego vehicle notice first" in question:
        return "notice_graph"
    if "lead to a collision" in question:
        return "collision_reasoning"
    if "safe actions" in question:
        return "safe_actions"
    if "what actions could the ego vehicle take" in question:
        return "candidate_actions"
    if question.startswith(("is ", "are ", "does ", "do ", "can ", "will ")):
        return "binary_reasoning"
    return f"{record[task]}_other"


def object_keys(text: str) -> set[tuple[str, str]]:
    return {(obj.lower(), camera.upper()) for obj, camera in OBJECT_ID_RE.findall(text)}


def summarize(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    if not rows:
        return {"count": 0}
    return {
        "count": len(rows),
        "exact_match": sum(row["exact"] for row in rows) / len(rows),
        "token_f1": sum(row["token_f1"] for row in rows) / len(rows),
        "rouge_l": sum(row["rouge_l"] for row in rows) / len(rows),
        "mean_reference_tokens": sum(row["reference_tokens"] for row in rows) / len(rows),
        "mean_prediction_tokens": sum(row["prediction_tokens"] for row in rows) / len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references-jsonl", required=True)
    parser.add_argument("--predictions-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--worst-per-family", type=int, default=3)
    args = parser.parse_args()

    references = read_jsonl(args.references_jsonl)
    predictions = {
        item["id"]: str(item["answer"])
        for item in json.loads(Path(args.predictions_json).read_text(encoding="utf-8"))
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    object_tp = object_pred = object_ref = 0
    for record in references:
        if record["id"] not in predictions:
            continue
        question = message_text(record, "user")
        reference = message_text(record, "assistant")
        prediction = predictions[record["id"]]
        ref_objects, pred_objects = object_keys(reference), object_keys(prediction)
        object_tp += len(ref_objects & pred_objects)
        object_ref += len(ref_objects)
        object_pred += len(pred_objects)
        row = {
            "id": record["id"],
            "task": record["task"],
            "question": question,
            "reference": reference,
            "prediction": prediction,
            "exact": float(normalize(prediction) == normalize(reference)),
            "token_f1": token_f1(prediction, reference),
            "rouge_l": rouge_l(prediction, reference),
            "reference_tokens": len(normalize(reference).split()),
            "prediction_tokens": len(normalize(prediction).split()),
        }
        grouped[question_family(record)].append(row)

    family_metrics = {
        family: summarize(rows)
        for family, rows in sorted(grouped.items())
    }
    worst_examples = {
        family: [
            {
                key: row[key]
                for key in ("id", "task", "question", "reference", "prediction", "token_f1")
            }
            for row in sorted(rows, key=lambda item: item["token_f1"])[:args.worst_per_family]
        ]
        for family, rows in sorted(grouped.items())
    }
    precision = object_tp / object_pred if object_pred else 0.0
    recall = object_tp / object_ref if object_ref else 0.0
    report = {
        "matched_predictions": sum(len(rows) for rows in grouped.values()),
        "family_metrics": family_metrics,
        "object_id_structure": {
            "true_positive": object_tp,
            "predicted": object_pred,
            "reference": object_ref,
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        },
        "worst_examples": worst_examples,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "worst_examples"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
