#!/usr/bin/env python3
"""Build deterministic high-confidence B10 preference pairs without external APIs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
COMMON_DIR = HERE.parent / "qwen_vl"
sys.path.insert(0, str(COMMON_DIR))
from common import message_text, read_jsonl  # noqa: E402


TOKEN_RE = re.compile(r"[a-z0-9]+|<[^>]+>", flags=re.IGNORECASE)
COORD_RE = re.compile(
    r"<[^,>]+,\s*[^,>]+,\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)>",
    flags=re.IGNORECASE,
)
ACTION_TERMS = {
    "stop": ("stop", "stopping", "remain stopped"),
    "brake": ("brake", "slow down", "decelerate"),
    "accelerate": ("accelerate", "speed up"),
    "straight": ("go straight", "continue straight", "drive straight"),
    "left": ("turn left", "change lane to the left", "move left"),
    "right": ("turn right", "change lane to the right", "move right"),
    "yield": ("yield", "give way"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--dev-jsonl", required=True)
    parser.add_argument("--candidate-jsonl", action="append", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--max-soft-similarity", type=float, default=0.80)
    return parser.parse_args()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def token_f1(reference: str, candidate: str) -> float:
    reference_tokens = tokens(reference)
    candidate_tokens = tokens(candidate)
    if not reference_tokens or not candidate_tokens:
        return float(reference_tokens == candidate_tokens)
    overlap = sum((Counter(reference_tokens) & Counter(candidate_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(candidate_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def lcs_length(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[-1], previous[index]))
        previous = current
    return previous[-1]


def rouge_l_f1(reference: str, candidate: str) -> float:
    reference_tokens = tokens(reference)
    candidate_tokens = tokens(candidate)
    if not reference_tokens or not candidate_tokens:
        return float(reference_tokens == candidate_tokens)
    overlap = lcs_length(reference_tokens, candidate_tokens)
    precision = overlap / len(candidate_tokens)
    recall = overlap / len(reference_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def coordinate_f1(reference: str, candidate: str) -> float:
    reference_coords = {
        (round(float(x), 1), round(float(y), 1)) for x, y in COORD_RE.findall(reference)
    }
    candidate_coords = {
        (round(float(x), 1), round(float(y), 1)) for x, y in COORD_RE.findall(candidate)
    }
    if not reference_coords or not candidate_coords:
        return float(reference_coords == candidate_coords)
    overlap = len(reference_coords & candidate_coords)
    precision = overlap / len(candidate_coords)
    recall = overlap / len(reference_coords)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def soft_similarity(reference: str, candidate: str) -> float:
    language = 0.5 * token_f1(reference, candidate) + 0.5 * rouge_l_f1(reference, candidate)
    if COORD_RE.search(reference) or COORD_RE.search(candidate):
        return 0.65 * coordinate_f1(reference, candidate) + 0.35 * language
    return language


def multiple_choice_letter(text: str) -> str | None:
    match = re.fullmatch(r"\s*([A-D])(?:[.):\s]*)", text, flags=re.IGNORECASE)
    return match.group(1).upper() if match else None


def binary_answer(text: str) -> str | None:
    normalized = text.strip().lower().rstrip(".")
    return normalized if normalized in {"yes", "no"} else None


def action_set(text: str) -> set[str]:
    normalized = text.lower()
    return {
        action
        for action, phrases in ACTION_TERMS.items()
        if any(phrase in normalized for phrase in phrases)
    }


def classify_pair(record: dict[str, Any], chosen: str, rejected: str, limit: float) -> tuple[str, float] | None:
    if chosen.strip() == rejected.strip():
        return None
    chosen_mc = multiple_choice_letter(chosen)
    rejected_mc = multiple_choice_letter(rejected)
    if chosen_mc and rejected_mc and chosen_mc != rejected_mc:
        return "multiple_choice_mismatch", 1.0
    chosen_binary = binary_answer(chosen)
    rejected_binary = binary_answer(rejected)
    if chosen_binary and rejected_binary and chosen_binary != rejected_binary:
        return "binary_mismatch", 1.0
    tags = {int(tag) for tag in record.get("tag", [])}
    if tags & {2, 3}:
        similarity = soft_similarity(chosen, rejected)
        if similarity <= limit:
            return "grounding_low_similarity", 1.0 - similarity
    if 1 in tags or str(record.get("task", "")).lower() == "planning":
        chosen_actions = action_set(chosen)
        rejected_actions = action_set(rejected)
        if chosen_actions and rejected_actions and chosen_actions.isdisjoint(rejected_actions):
            return "planning_action_conflict", 1.0
    return None


def read_candidates(paths: list[str]) -> dict[str, list[str]]:
    candidates_by_id: dict[str, list[str]] = {}
    for candidate_path in paths:
        with Path(candidate_path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                item = json.loads(line)
                item_id = str(item["id"])
                if item_id in candidates_by_id:
                    raise ValueError(f"Duplicate candidate id {item_id} in {candidate_path}:{line_number}")
                candidates_by_id[item_id] = [str(value).strip() for value in item["candidates"] if str(value).strip()]
    return candidates_by_id


def main() -> None:
    args = parse_args()
    train_records = read_jsonl(args.train_jsonl)
    dev_records = read_jsonl(args.dev_jsonl)
    train_scenes = {str(record["scene_id"]) for record in train_records}
    dev_scenes = {str(record["scene_id"]) for record in dev_records}
    overlap = train_scenes & dev_scenes
    if overlap:
        raise ValueError(f"Scene leakage detected between train and dev: {len(overlap)} scenes")
    train_by_id = {str(record["id"]): record for record in train_records}
    dev_ids = {str(record["id"]) for record in dev_records}
    candidates_by_id = read_candidates(args.candidate_jsonl)
    candidate_ids = set(candidates_by_id)
    unexpected = candidate_ids - set(train_by_id)
    if unexpected:
        raise ValueError(f"Candidate files contain {len(unexpected)} non-train ids")

    preference_rows: list[dict[str, Any]] = []
    by_task: Counter[str] = Counter()
    by_tag: Counter[str] = Counter()
    by_reason: Counter[str] = Counter()
    candidate_records_without_pair = 0
    for item_id, candidates in candidates_by_id.items():
        record = train_by_id[item_id]
        chosen = message_text(record, "assistant").strip()
        accepted: tuple[str, str, float] | None = None
        for rejected in candidates:
            decision = classify_pair(record, chosen, rejected, args.max_soft_similarity)
            if decision is None:
                continue
            reason, margin = decision
            if accepted is None or margin > accepted[2]:
                accepted = (rejected, reason, margin)
        if accepted is None:
            candidate_records_without_pair += 1
            continue
        rejected, reason, margin = accepted
        row = {
            "id": item_id,
            "scene_id": str(record["scene_id"]),
            "frame_id": str(record["frame_id"]),
            "qa_index": int(record["qa_index"]),
            "task": str(record["task"]),
            "tag": record["tag"],
            "images": record["images"],
            "system": message_text(record, "system"),
            "question": message_text(record, "user"),
            "chosen": chosen,
            "rejected": rejected,
            "selection_reason": reason,
            "selection_margin": margin,
            "policy_source": "B10",
        }
        preference_rows.append(row)
        by_task[row["task"]] += 1
        by_reason[reason] += 1
        for tag in row["tag"]:
            by_tag[str(tag)] += 1

    output = Path(args.output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in preference_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "schema_version": "drivelm-v037a-preferences-v1",
        "objective": "train-only deterministic high-confidence offline preference pairs",
        "train_records": len(train_records),
        "dev_records": len(dev_records),
        "train_scenes": len(train_scenes),
        "dev_scenes": len(dev_scenes),
        "train_dev_scene_overlap": len(overlap),
        "candidate_records": len(candidates_by_id),
        "candidate_ids_outside_train": len(unexpected),
        "preference_pairs": len(preference_rows),
        "candidate_records_without_pair": candidate_records_without_pair,
        "dev_ids_in_preferences": sum(row["id"] in dev_ids for row in preference_rows),
        "by_task": dict(sorted(by_task.items())),
        "by_tag": dict(sorted(by_tag.items())),
        "by_reason": dict(sorted(by_reason.items())),
        "max_soft_similarity": args.max_soft_similarity,
        "selection": "deterministic rules only; ambiguous free-form candidates excluded",
        "semantic_api_used": False,
        "train_manifest_sha256": sha256(args.train_jsonl),
        "dev_manifest_sha256": sha256(args.dev_jsonl),
        "candidate_files": [
            {"path": str(Path(path).resolve()), "sha256": sha256(path)}
            for path in args.candidate_jsonl
        ],
        "output_sha256": sha256(output),
    }
    if report["dev_ids_in_preferences"] != 0:
        raise AssertionError("Dev leakage detected in preference output")
    report_path = output.with_suffix(output.suffix + ".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "report": str(report_path), **report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
