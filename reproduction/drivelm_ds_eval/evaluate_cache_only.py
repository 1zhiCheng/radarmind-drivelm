#!/usr/bin/env python3
"""Run DriveLM-DS using only previously cached semantic-judge results.

This wrapper deliberately never constructs an API client. It preserves the exact
cache key used by ``deepseek_judge.py`` and raises on the first cache miss.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import evaluate as evaluator
from deepseek_judge import JudgeResult, PROMPT_VERSION


class CacheOnlyJudge:
    def __init__(
        self,
        *,
        secret_file: Path,
        cache_file: Path,
        model: str = "deepseek-v4-flash",
        base_url: str = "https://api.deepseek.com",
        **_: object,
    ) -> None:
        del secret_file
        self.cache_file = cache_file
        self.model = model
        self.base_url = base_url.rstrip("/")

    def _cache_key(self, kind: str, candidate: str, reference: str) -> str:
        payload = {
            "prompt_version": PROMPT_VERSION,
            "model": self.model,
            "base_url": self.base_url,
            "kind": kind,
            "candidate": candidate,
            "reference": reference,
            "temperature": 0,
            "thinking": "disabled",
        }
        return hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def score(self, kind: str, candidate: str, reference: str) -> JudgeResult:
        key = self._cache_key(kind, candidate, reference)
        uri = f"file:{self.cache_file.resolve()}?mode=ro"
        with sqlite3.connect(uri, uri=True, timeout=30) as connection:
            row = connection.execute(
                "SELECT payload FROM judge_scores WHERE cache_key = ?", (key,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Judge cache miss in network-disabled evaluator: {key}")
        payload = json.loads(row[0])
        payload["cache_hit"] = True
        return JudgeResult(**payload)


if __name__ == "__main__":
    evaluator.DeepSeekJudge = CacheOnlyJudge
    evaluator.main()
