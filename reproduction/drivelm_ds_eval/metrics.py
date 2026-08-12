#!/usr/bin/env python3
"""Deterministic portions of the public DriveLM challenge protocol."""

from __future__ import annotations

import re
from dataclasses import dataclass


# Kept intentionally compatible with challenge/evaluation.py: integers and
# negative values are not treated as coordinates by the public evaluator.
FLOAT_RE = re.compile(r"\d+\.\d+")


def coordinate_pairs(text: str) -> list[tuple[float, float]]:
    values = [float(value) for value in FLOAT_RE.findall(text)]
    if len(values) % 2:
        values = values[:-1]
    return list(zip(values[::2], values[1::2]))


@dataclass(frozen=True)
class CoordinateMatch:
    matched: tuple[tuple[float, float], ...]
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float


def match_coordinates(candidate: str, reference: str, threshold: float = 16.0) -> CoordinateMatch:
    predictions = coordinate_pairs(candidate)
    remaining = list(coordinate_pairs(reference))
    reference_count = len(remaining)
    matched: list[tuple[float, float]] = []
    true_positives = 0
    false_positives = 0
    for prediction in predictions:
        if not remaining:
            false_positives += 1
            continue
        distances = [
            abs(prediction[0] - target[0]) + abs(prediction[1] - target[1])
            for target in remaining
        ]
        closest_index = min(range(len(distances)), key=distances.__getitem__)
        if distances[closest_index] < threshold:
            true_positives += 1
            matched.append(remaining.pop(closest_index))
        else:
            false_positives += 1
    false_negatives = reference_count - true_positives
    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return CoordinateMatch(
        matched=tuple(matched),
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def graph_question_is_eligible(question: str, matched_graph: tuple[tuple[float, float], ...]) -> bool:
    return all(pair in matched_graph for pair in coordinate_pairs(question))


def combine_language(language: dict[str, float]) -> float:
    bleu = sum(language[f"Bleu_{index}"] for index in range(1, 5)) / 4.0
    return (bleu + language["ROUGE_L"] + language["CIDEr"] / 10.0) / 3.0


def combine_final(
    *, accuracy: float, planning_judge_100: float, language: float, match_100: float
) -> float:
    return (
        0.2 * accuracy
        + 0.4 * planning_judge_100 / 100.0
        + 0.2 * language
        + 0.2 * match_100 / 100.0
    )
