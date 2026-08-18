# v0.43 InternVL-inspired fixed anchor ensemble

This experiment transfers the complementary-ensemble observation from
InternVL4Drive without retraining or overwriting either source model.

## Source predictions

- downstream baseline: v0.39B MoL checkpoint 700;
- frame-anchor expert: v0.42A Graph-SFT checkpoint 600;
- references: the frozen 3,355-QA scene-isolated dev JSONL.

## Build the candidate

```bash
python reproduction/qwen_vl_v043_internvl_transfer/build_fixed_ensemble.py \
  --references-jsonl /mnt/data/zzy/drivelm/reproduction/qwen_vl_stage2_ablation_v1/qwen_v033_c00_dev.jsonl \
  --baseline-json /mnt/data/zzy/drivelm/reproduction/qwen_vl_v039_mol/sweep/checkpoint-700/mol_dev_predictions.json \
  --grounding-json /mnt/data/zzy/drivelm/reproduction/qwen_vl_v042_graph/rollout_checkpoint600/graph_dev_predictions.json \
  --mode anchor \
  --output-json /mnt/data/zzy/drivelm/reproduction/qwen_vl_v043_internvl_transfer/anchor/ensemble_predictions.json \
  --report-json /mnt/data/zzy/drivelm/reproduction/qwen_vl_v043_internvl_transfer/anchor/build_report.json
```

The rule is fixed before evaluation: use Graph-A only when `qa_index == 0`,
otherwise use MoL-700. It routes 467/3,355 answers to Graph-A.

## Network-disabled evaluation

```bash
python reproduction/drivelm_ds_eval/evaluate_cache_only.py \
  --references-jsonl /mnt/data/zzy/drivelm/reproduction/qwen_vl_stage2_ablation_v1/qwen_v033_c00_dev.jsonl \
  --predictions-json /mnt/data/zzy/drivelm/reproduction/qwen_vl_v043_internvl_transfer/anchor/ensemble_predictions.json \
  --output-json /mnt/data/zzy/drivelm/reproduction/qwen_vl_v043_internvl_transfer/anchor/drivelm_ds.json \
  --cache-file /mnt/data/zzy/drivelm/reproduction/drivelm_ds/deepseek_judge.sqlite \
  --workers 8
```

`evaluate_cache_only.py` never constructs an API client and fails on a cache
miss. The recorded run completes 791/791 cached items and obtains Final
`0.6122931502241084`.

## Tests

```bash
pytest -q reproduction/qwen_vl_v043_internvl_transfer/test_fixed_ensemble.py
```

See `docs/current/VERSION_0_43_DRIVELM_INTERNVL_TRANSFER_ENSEMBLE.md` and
`reports/v043_internvl_transfer/technical_report.pdf` for the full audit.
