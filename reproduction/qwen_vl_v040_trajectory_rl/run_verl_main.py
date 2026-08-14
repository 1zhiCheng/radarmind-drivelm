#!/usr/bin/env python3
"""Run the VERL Hydra entrypoint and release this driver's Ray runtime on exit."""

from __future__ import annotations

import runpy

import ray


try:
    runpy.run_module("verl.trainer.main_ppo", run_name="__main__")
finally:
    # Only disconnect the runtime created by this training driver; never issue
    # a machine-wide ``ray stop`` that could affect unrelated jobs.
    if ray.is_initialized():
        ray.shutdown()
