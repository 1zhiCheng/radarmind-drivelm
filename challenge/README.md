# DriveLM challenge compatibility tools

This directory retains the public DriveLM conversion and evaluation utilities
required by the RadarMind-DriveLM dataset builder.

## Included

- `extract_data.py`: extracts official frame-level QA records.
- `convert_data.py`: builds the public evaluation representation.
- `convert2llama.py`: preserves the original deterministic multiple-choice conversion.
- `evaluation.py`: public metric structure and graph-gating reference.
- `prepare_submission.py`: validates the official `id + answer` output format.
- `gpt_eval.py`: upstream semantic-judge reference only.

The legacy LLaMA-Adapter baseline and generated test outputs are intentionally
not mirrored because the active repository uses Qwen2.5-VL. See the
[root quick start](../README.md) for the supported pipeline.

These files originate from the Apache-2.0 DriveLM release. When using the
dataset or evaluation protocol, cite the original DriveLM project.
