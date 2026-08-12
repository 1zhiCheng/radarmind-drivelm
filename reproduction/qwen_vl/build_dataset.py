#!/usr/bin/env python3
"""Build deterministic DriveLM challenge and six-camera Qwen-VL datasets."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


CAMERA_ORDER = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)
TASK_ORDER = ("perception", "prediction", "planning", "behavior")
SYSTEM_PROMPT = (
    "You are an autonomous-driving perception and decision assistant. Analyze all six "
    "synchronized surround-view camera images. Answer the driving question accurately "
    "and concisely. Do not invent objects that are not visible."
)


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def scene_is_dev(scene_id: str, dev_ratio: float) -> bool:
    bucket = int(hashlib.sha256(scene_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return bucket < dev_ratio


def resolve_images(image_paths: dict[str, str], annotation_path: Path) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for camera in CAMERA_ORDER:
        raw = Path(image_paths[camera])
        path = raw if raw.is_absolute() else annotation_path.parent / raw
        resolved[camera] = str(path.resolve())
    return resolved


def iter_records(
    data: dict[str, Any],
    annotation_path: Path,
    *,
    require_answer: bool,
) -> Iterable[dict[str, Any]]:
    for scene_id, scene in data.items():
        for frame_id, frame in scene["key_frames"].items():
            images = resolve_images(frame["image_paths"], annotation_path)
            qas: list[tuple[str, dict[str, Any]]] = []
            for task in TASK_ORDER:
                qas.extend((task, qa) for qa in frame["QA"].get(task, []))
            for index, (task, qa) in enumerate(qas):
                answer = qa.get("A")
                has_answer = answer is not None and str(answer).strip() != ""
                if require_answer and not has_answer:
                    continue
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": qa["Q"]},
                ]
                if has_answer:
                    messages.append({"role": "assistant", "content": str(answer)})
                yield {
                    "id": f"{scene_id}_{frame_id}_{index}",
                    "scene_id": scene_id,
                    "frame_id": frame_id,
                    "qa_index": index,
                    "task": task,
                    "tag": qa.get("tag"),
                    "images": images,
                    "messages": messages,
                }


def validate_images(records: list[dict[str, Any]]) -> tuple[int, list[str]]:
    missing: set[str] = set()
    unique: set[str] = set()
    for record in records:
        for path in record["images"].values():
            unique.add(path)
            if not Path(path).is_file():
                missing.add(path)
    return len(unique), sorted(missing)


def run_official_conversion(train_path: Path, output_dir: Path, seed: int) -> tuple[Path, Path, Path]:
    repo_root = Path(__file__).resolve().parents[2]
    challenge_dir = repo_root / "challenge"
    sys.path.insert(0, str(challenge_dir))
    from convert2llama import convert2llama
    from convert_data import loop_test
    from extract_data import extract_data

    extracted = output_dir / "official_extracted.json"
    evaluation = output_dir / "official_train_eval.json"
    llama = output_dir / "official_train_llama.json"
    log_path = output_dir / "official_extract.log"
    random.seed(seed)
    with log_path.open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
        extract_data(str(train_path), str(extracted))
    random.seed(seed)
    loop_test(str(extracted), str(evaluation))
    convert2llama(str(evaluation), str(llama))
    return extracted, evaluation, llama


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--val-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--skip-official", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.skip_official:
        official_eval = args.output_dir / "official_train_eval.json"
        if not official_eval.is_file():
            raise FileNotFoundError(f"Missing {official_eval}; remove --skip-official for the first run")
    else:
        _, official_eval, _ = run_official_conversion(args.train_json, args.output_dir, args.seed)

    official_data = json.loads(official_eval.read_text(encoding="utf-8"))
    train_records: list[dict[str, Any]] = []
    dev_records: list[dict[str, Any]] = []
    for record in iter_records(official_data, args.train_json, require_answer=True):
        (dev_records if scene_is_dev(record["scene_id"], args.dev_ratio) else train_records).append(record)

    val_data = json.loads(args.val_json.read_text(encoding="utf-8"))
    val_records = list(iter_records(val_data, args.val_json, require_answer=False))
    unique_images, missing = validate_images(train_records + dev_records + val_records)
    if missing:
        sample = "\n".join(missing[:10])
        raise FileNotFoundError(f"{len(missing)} referenced images are missing. First paths:\n{sample}")

    counts = {
        "train": write_jsonl(args.output_dir / "qwen_train.jsonl", train_records),
        "dev": write_jsonl(args.output_dir / "qwen_dev.jsonl", dev_records),
        "val_questions": write_jsonl(args.output_dir / "qwen_val_questions.jsonl", val_records),
    }
    report = {
        "train_annotation": str(args.train_json.resolve()),
        "val_annotation": str(args.val_json.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "seed": args.seed,
        "dev_ratio": args.dev_ratio,
        "camera_order": list(CAMERA_ORDER),
        "counts": counts,
        "train_task_counts": dict(Counter(x["task"] for x in train_records)),
        "dev_task_counts": dict(Counter(x["task"] for x in dev_records)),
        "val_task_counts": dict(Counter(x["task"] for x in val_records)),
        "train_scenes": len({x["scene_id"] for x in train_records}),
        "dev_scenes": len({x["scene_id"] for x in dev_records}),
        "val_scenes": len({x["scene_id"] for x in val_records}),
        "unique_images": unique_images,
        "missing_images": 0,
    }
    dump_json(args.output_dir / "dataset_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
