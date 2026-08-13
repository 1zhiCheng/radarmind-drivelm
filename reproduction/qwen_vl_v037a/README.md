# DriveLM v0.37A B10-DPO

This directory contains the train-only preference construction, frozen B10 reference scoring, three-GPU DPO, sharded inference and promotion-gate utilities used by v0.37A.

## Outcome

- 26,095 sampled train records produced 7,149 high-confidence pairs.
- Train/dev scene overlap and dev IDs in preference data were both zero.
- Three-GPU DPO completed 596 steps.
- The final adapter improved MC accuracy but reduced DriveLM-DS Final from 0.59464 to 0.55330.
- Checkpoints 100, 200 and 300 also failed to exceed B10; B10 remains the promoted model.

See docs/current/VERSION_0_37A_DRIVELM_B10_DPO.md for the full configuration, checkpoint sweep and next-stage design.

## Components

- generate_candidates.py: resumable sampled B10 candidates with deterministic GPU sharding.
- build_preferences.py and audit_preferences.py: high-confidence train-only pairs and leakage gates.
- precompute_reference.py and merge_reference_shards.py: frozen B10 log-probabilities.
- train_dpo_ddp.py: memory-efficient three-GPU DPO.
- shard_inference.py: deterministic parallel dev inference and strict merge.
- compare_results.py: B10 promotion gates.
- run_full_pipeline.sh: server-specific orchestration example; update paths and GPU UUIDs before use.
