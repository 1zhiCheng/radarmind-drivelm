#!/usr/bin/env python3
"""Deterministic, API-free planning reward shared by GRPO and GSPO."""

from __future__ import annotations

import re
from collections import Counter


OBJECT_RE = re.compile(r"<c\d+,[^>]+>", re.IGNORECASE)
ACTION_PATTERNS = {
    "brake": r"\b(brake|emergency stop|come to a stop|stop the vehicle)\b",
    "slow": r"\b(slow down|decelerate|reduce speed)\b",
    "accelerate": r"\b(accelerate|speed up)\b",
    "left": r"\b(turn left|steer left)\b",
    "right": r"\b(turn right|steer right)\b",
    "straight": r"\b(go straight|keep straight|continue straight)\b",
    "lane_change": r"\b(change lanes?|switch lanes?)\b",
    "yield": r"\b(yield|give way)\b",
    "wait": r"\b(wait|remain stationary)\b",
    "proceed": r"\b(proceed|move forward|go ahead|continue driving)\b",
    "reverse": r"\b(reverse|back up)\b",
    "overtake": r"\b(overtake|pass the)\b",
}


def normalize(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9<>.,]+", " ", text.lower()).split())


def token_f1(prediction: str, reference: str) -> float:
    pred, ref = normalize(prediction).split(), normalize(reference).split()
    if not pred or not ref:
        return float(pred == ref)
    overlap = sum((Counter(pred) & Counter(ref)).values())
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
        for index, target in enumerate(ref, 1):
            row[index] = previous[index - 1] + 1 if token == target else max(previous[index], row[index - 1])
    lcs = row[-1]
    if not lcs:
        return 0.0
    precision, recall = lcs / len(pred), lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


def action_set(text: str) -> set[str]:
    normalized = normalize(text)
    return {name for name, pattern in ACTION_PATTERNS.items() if re.search(pattern, normalized)}


def set_f1(prediction: set[str], reference: set[str]) -> float:
    if not reference:
        return 1.0 if not prediction else 0.0
    overlap = len(prediction & reference)
    if not overlap:
        return 0.0
    precision, recall = overlap / max(len(prediction), 1), overlap / len(reference)
    return 2 * precision * recall / (precision + recall)


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **_kwargs):
    if data_source != "radarmind_drivelm_trajectory_planning":
        raise ValueError(f"unsupported data source: {data_source}")
    solution, reference = str(solution_str).strip(), str(ground_truth).strip()
    info = extra_info or {}
    lexical = token_f1(solution, reference)
    rouge = rouge_l(solution, reference)
    exact = float(normalize(solution) == normalize(reference))
    action = set_f1(action_set(solution), action_set(reference))
    predicted_objects = set(OBJECT_RE.findall(solution))
    allowed_objects = set(info.get("allowed_object_ids", []))
    grounding = (
        0.0 if not solution
        else 1.0 if not predicted_objects
        else len(predicted_objects & allowed_objects) / len(predicted_objects)
    )
    word_count = len(solution.split())
    format_reward = float(0 < word_count <= 256 and "<TRAJECTORY_STATE" not in solution)
    score = (
        0.40 * lexical
        + 0.20 * rouge
        + 0.25 * action
        + 0.05 * exact
        + 0.05 * grounding
        + 0.05 * format_reward
    )
    return {
        "score": float(max(0.0, min(score, 1.0))),
        "token_f1_reward": lexical,
        "rouge_l_reward": rouge,
        "action_f1_reward": action,
        "exact_reward": exact,
        "grounding_reward": grounding,
        "format_reward": format_reward,
    }
