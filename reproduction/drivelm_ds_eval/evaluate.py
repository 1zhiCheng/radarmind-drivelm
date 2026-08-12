#!/usr/bin/env python3
"""Evaluate labeled DriveLM JSONL with public metrics and DeepSeek judge.

This produces a local proxy named DriveLM-DS. It is not the hidden challenge
server score because the public GPT-3.5 judge is replaced by DeepSeek-V4-Flash.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from deepseek_judge import DeepSeekJudge, JudgeResult, PROMPT_VERSION
from metrics import (
    combine_final,
    combine_language,
    graph_question_is_eligible,
    match_coordinates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references-jsonl", type=Path, required=True)
    parser.add_argument("--predictions-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--secret-file", type=Path,
        default=Path.home() / ".config" / "radarmind" / "deepseek_api_key",
    )
    parser.add_argument("--cache-file", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--judge-limit", type=int, default=0,
        help="Only judge N semantic items for an API smoke test; suppresses final score.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def message(record: dict[str, Any], role: str) -> str:
    for item in record["messages"]:
        if item["role"] == role:
            return str(item["content"])
    raise KeyError(f"{record['id']} has no {role} message")


def language_metrics(candidates: list[str], references: list[str]) -> dict[str, float]:
    try:
        import language_evaluation
    except ImportError as error:
        raise RuntimeError(
            "language_evaluation is required for public BLEU/ROUGE_L/CIDEr metrics"
        ) from error
    evaluator = language_evaluation.CocoEvaluator(coco_types=["BLEU", "ROUGE_L", "CIDEr"])
    raw = evaluator.run_evaluation(candidates, references)
    expected = ("Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "ROUGE_L", "CIDEr")
    return {key: float(raw[key]) for key in expected}


def judge_many(
    judge: DeepSeekJudge,
    jobs: list[tuple[str, str, str, str]],
    workers: int,
) -> tuple[dict[str, JudgeResult], list[dict[str, str]]]:
    results: dict[str, JudgeResult] = {}
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(judge.score, kind, candidate, reference): (record_id, kind)
            for record_id, kind, candidate, reference in jobs
        }
        for future in as_completed(futures):
            record_id, kind = futures[future]
            try:
                results[record_id] = future.result()
            except Exception as error:
                failures.append({"id": record_id, "kind": kind, "error": repr(error)})
    return results, failures


def main() -> None:
    args = parse_args()
    if args.workers <= 0 or args.judge_limit < 0:
        raise ValueError("workers must be positive and judge-limit non-negative")
    records = read_jsonl(args.references_jsonl)
    prediction_rows = json.loads(args.predictions_json.read_text(encoding="utf-8"))
    predictions = {str(row["id"]): str(row["answer"]) for row in prediction_rows}
    reference_ids = {str(record["id"]) for record in records}
    missing_ids = sorted(reference_ids - predictions.keys())
    extra_ids = sorted(predictions.keys() - reference_ids)

    frames: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        frames[(record["scene_id"], record["frame_id"])].append(record)

    eligible: list[dict[str, Any]] = []
    gated_out: list[str] = []
    coordinate_rows: list[dict[str, Any]] = []
    for frame_records in frames.values():
        ordered = sorted(frame_records, key=lambda item: int(item["qa_index"]))
        first = ordered[0]
        first_prediction = predictions.get(first["id"], "")
        first_reference = message(first, "assistant")
        graph_match = match_coordinates(first_prediction, first_reference)
        matched_graph = graph_match.matched
        for index, record in enumerate(ordered):
            if record["id"] not in predictions:
                continue
            question = message(record, "user")
            if index > 0 and not graph_question_is_eligible(question, matched_graph):
                gated_out.append(record["id"])
                continue
            eligible.append(record)
            if 3 in record["tag"]:
                match = match_coordinates(predictions[record["id"]], message(record, "assistant"))
                coordinate_rows.append(
                    {
                        "id": record["id"],
                        "precision": match.precision,
                        "recall": match.recall,
                        "f1": match.f1,
                        "tp": match.true_positives,
                        "fp": match.false_positives,
                        "fn": match.false_negatives,
                    }
                )

    by_tag = Counter(tag for record in eligible for tag in record["tag"])
    accuracy_records = [record for record in eligible if 0 in record["tag"]]
    language_records = [record for record in eligible if 2 in record["tag"]]
    judge_records = [record for record in eligible if 1 in record["tag"] or 3 in record["tag"]]
    jobs = [
        (
            record["id"],
            "planning" if 1 in record["tag"] else "graph",
            predictions[record["id"]],
            message(record, "assistant"),
        )
        for record in judge_records
    ]
    if args.judge_limit:
        jobs = jobs[:args.judge_limit]

    judge = DeepSeekJudge(
        secret_file=args.secret_file,
        cache_file=args.cache_file,
        model=args.model,
        base_url=args.base_url,
    )
    judge_results, judge_failures = judge_many(judge, jobs, args.workers)

    accuracy = (
        sum(predictions[record["id"]] == message(record, "assistant") for record in accuracy_records)
        / max(len(accuracy_records), 1)
    )
    language_raw = language_metrics(
        [predictions[record["id"]] for record in language_records],
        [message(record, "assistant") for record in language_records],
    )
    language_combined = combine_language(language_raw)
    coordinate_f1 = sum(row["f1"] for row in coordinate_rows) / max(len(coordinate_rows), 1)

    planning_scores = [
        judge_results[record["id"]].score
        for record in eligible if 1 in record["tag"] and record["id"] in judge_results
    ]
    graph_scores = [
        judge_results[record["id"]].score
        for record in eligible if 3 in record["tag"] and record["id"] in judge_results
    ]
    complete_judging = len(judge_results) == len(judge_records) and not judge_failures
    planning_judge = sum(planning_scores) / max(len(planning_scores), 1)
    graph_judge = sum(graph_scores) / max(len(graph_scores), 1)
    match_score = (coordinate_f1 * 100.0 + graph_judge) / 2.0
    final_score = (
        combine_final(
            accuracy=accuracy,
            planning_judge_100=planning_judge,
            language=language_combined,
            match_100=match_score,
        )
        if complete_judging and not args.judge_limit and not missing_ids
        else None
    )

    report = {
        "protocol": {
            "name": "DriveLM-DS",
            "description": "Public DriveLM metric structure with DeepSeek semantic judge; not hidden-server score.",
            "judge_model": args.model,
            "judge_prompt_version": PROMPT_VERSION,
            "thinking": "disabled",
            "temperature": 0,
            "graph_gating": "public challenge/evaluation.py compatible",
        },
        "inputs": {
            "references": str(args.references_jsonl.resolve()),
            "predictions": str(args.predictions_json.resolve()),
            "reference_count": len(records),
            "prediction_count": len(predictions),
            "missing_count": len(missing_ids),
            "extra_count": len(extra_ids),
            "missing_ids": missing_ids[:50],
            "extra_ids": extra_ids[:50],
        },
        "gating": {
            "eligible_count": len(eligible),
            "gated_out_count": len(gated_out),
            "eligible_by_tag": {str(key): value for key, value in sorted(by_tag.items())},
        },
        "judge": {
            "required": len(judge_records),
            "requested": len(jobs),
            "completed": len(judge_results),
            "cache_hits": sum(result.cache_hit for result in judge_results.values()),
            "failures": judge_failures,
            "complete": complete_judging,
            "smoke_limit": args.judge_limit,
            "sample_results": {
                record_id: {
                    "score": result.score,
                    "semantic_correctness": result.semantic_correctness,
                    "action_correctness": result.action_correctness,
                    "object_state_correctness": result.object_state_correctness,
                    "format_valid": result.format_valid,
                    "brief_reason": result.brief_reason,
                    "cache_hit": result.cache_hit,
                    "system_fingerprint": result.system_fingerprint,
                }
                for record_id, result in list(judge_results.items())[:20]
            },
        },
        "metrics": {
            "accuracy": accuracy,
            "accuracy_count": len(accuracy_records),
            "planning_deepseek_100": planning_judge if planning_scores else None,
            "planning_judged_count": len(planning_scores),
            "language_raw": language_raw,
            "language_combined": language_combined,
            "language_count": len(language_records),
            "coordinate_precision_macro": sum(row["precision"] for row in coordinate_rows) / max(len(coordinate_rows), 1),
            "coordinate_recall_macro": sum(row["recall"] for row in coordinate_rows) / max(len(coordinate_rows), 1),
            "coordinate_f1_macro": coordinate_f1,
            "coordinate_count": len(coordinate_rows),
            "graph_deepseek_100": graph_judge if graph_scores else None,
            "graph_judged_count": len(graph_scores),
            "match_score_100": match_score if graph_scores else None,
            "drivelm_ds_final": final_score,
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output_json),
        "judge_completed": len(judge_results),
        "judge_failures": len(judge_failures),
        "cache_hits": report["judge"]["cache_hits"],
        "drivelm_ds_final": final_score,
    }, indent=2))


if __name__ == "__main__":
    main()
