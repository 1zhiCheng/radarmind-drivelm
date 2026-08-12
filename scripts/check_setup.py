#!/usr/bin/env python3
"""Fail-fast audit for the DriveLM annotations, six cameras and local model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CAMERAS = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--model-path", type=Path, default=Path("models/Qwen2.5-VL-3B-Instruct"))
    args = parser.parse_args()

    annotations = args.data_root / "QA_dataset_nus"
    required = [
        annotations / "v1_1_train_nus.json",
        annotations / "v1_1_val_nus_q_only.json",
        args.model_path / "config.json",
    ]
    camera_root = args.data_root / "nuscenes" / "samples"
    required.extend(camera_root / camera for camera in CAMERAS)
    missing = [str(path) for path in required if not path.exists()]
    report = {
        "data_root": str(args.data_root.resolve()),
        "model_path": str(args.model_path.resolve()),
        "six_camera_dirs": sum((camera_root / camera).is_dir() for camera in CAMERAS),
        "missing": missing,
        "ready": not missing,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
