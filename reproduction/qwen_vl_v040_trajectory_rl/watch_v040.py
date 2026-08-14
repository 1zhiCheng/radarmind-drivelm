#!/usr/bin/env python3
"""Supervise v0.40 trajectory RL and restart an exited pipeline."""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path("/home/zhangzongyuan/Myproject/drivelm/DriveLM-main")
RUN = Path("/mnt/data/zzy/drivelm/reproduction/qwen_vl_v040_trajectory_rl")
PIPELINE = ROOT / "reproduction/qwen_vl_v040_trajectory_rl/run_v040_pipeline.sh"
STATUS = RUN / "watchdog_status.json"
EVENTS = RUN / "logs/watchdog_events.log"
SESSION = "v040_trajectory"
INTERVAL_SECONDS = 15
MAX_SAME_FAILURE_RESTARTS = 3


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=False)


def session_alive() -> bool:
    return run(["tmux", "has-session", "-t", SESSION]).returncode == 0


def latest_log() -> Path | None:
    paths = [RUN / "logs" / name for name in (
        "train_gspo.log", "train_grpo.log", "smoke_gspo.log", "smoke_grpo.log"
    )]
    existing = [path for path in paths if path.exists()]
    return max(existing, key=lambda path: path.stat().st_mtime) if existing else None


def failure_signature(path: Path | None) -> str:
    if path is None:
        return "NO_LOG"
    text = path.read_text(errors="replace")[-200_000:]
    checks = [
        ("HOST_MEMORY_OOM", (
            "node running low on memory",
            "exceeded the memory usage threshold",
            "exceeded threshold of",
        )),
        ("CUDA_OUT_OF_MEMORY", ("CUDA out of memory", "torch.OutOfMemoryError")),
        ("VLLM_KV_CACHE", ("No available memory for the cache blocks",)),
        ("HYDRA_CONFIG", ("Could not override", "ConfigKeyError")),
        ("ASSERTION", ("AssertionError",)),
        ("VALUE_ERROR", ("ValueError",)),
        ("RUNTIME_ERROR", ("RuntimeError",)),
        ("TRACEBACK", ("Traceback",)),
    ]
    for name, needles in checks:
        if any(needle in text for needle in needles):
            return name
    return "PROCESS_EXITED"


def gpu_snapshot() -> list[dict[str, str]]:
    result = run([
        "nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ])
    rows = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 3)]
        if len(fields) == 4:
            rows.append(dict(zip(
                ("index", "name", "memory_mib", "utilization_pct"), fields
            )))
    return rows


def write_status(**values: object) -> None:
    payload = {
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        **values,
    }
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(STATUS)


def record(message: str) -> None:
    EVENTS.parent.mkdir(parents=True, exist_ok=True)
    with EVENTS.open("a") as stream:
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        stream.write(f"{timestamp} {message}\n")


def launch() -> None:
    command = f"cd {ROOT} && bash {PIPELINE}"
    result = run(["tmux", "new-session", "-d", "-s", SESSION, command])
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "tmux launch failed")


def main() -> int:
    previous_signature = ""
    same_failure_restarts = 0
    record("watchdog started")
    while True:
        if (RUN / "TRAINING_COMPLETE").exists():
            write_status(state="complete", gpu=gpu_snapshot())
            record("training complete; watchdog exiting")
            return 0

        log = latest_log()
        if session_alive():
            write_status(
                state="running",
                session=SESSION,
                latest_log=str(log) if log else None,
                latest_log_mtime=log.stat().st_mtime if log else None,
                gpu=gpu_snapshot(),
                same_failure_restarts=same_failure_restarts,
            )
        else:
            signature = failure_signature(log)
            same_failure_restarts = (
                same_failure_restarts + 1 if signature == previous_signature else 1
            )
            previous_signature = signature
            if same_failure_restarts > MAX_SAME_FAILURE_RESTARTS:
                write_status(
                    state="blocked_repeated_failure",
                    failure_signature=signature,
                    same_failure_restarts=same_failure_restarts - 1,
                    latest_log=str(log) if log else None,
                    gpu=gpu_snapshot(),
                )
                record(f"blocked after repeated failure: {signature}")
                return 2
            record(f"pipeline absent; restart {same_failure_restarts}: {signature}")
            launch()
            write_status(
                state="restarted",
                failure_signature=signature,
                same_failure_restarts=same_failure_restarts,
                latest_log=str(log) if log else None,
                gpu=gpu_snapshot(),
            )
        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
