# DriveLM-DS local evaluation

This evaluator keeps the public DriveLM challenge structure while replacing the
unavailable GPT-3.5 semantic judge with a frozen `deepseek-v4-flash` judge.
Consequently, `drivelm_ds_final` is a local proxy and must not be reported as the
hidden challenge-server score.

The deterministic components remain compatible with `challenge/evaluation.py`:

- exact-string Accuracy for tag 0;
- BLEU-1..4, ROUGE-L and CIDEr for tag 2;
- greedy Manhattan coordinate matching with distance `< 16` for tag 3;
- per-frame graph gating from the first important-object answer;
- public final weights 0.4 / 0.2 / 0.2 / 0.2.

DeepSeek scores tag-1 planning semantics and the semantic half of tag-3 Match.
Calls use non-thinking mode, temperature zero, JSON output and a SQLite cache.
The API key is read from `~/.config/radarmind/deepseek_api_key`
and is never included in an artifact.

## Smoke test

```bash
cd reproduction/drivelm_ds_eval
python evaluate.py \
  --references-jsonl ../../data/reproduction/qwen_vl/qwen_dev.jsonl \
  --predictions-json ../../outputs/dev_predictions.json \
  --output-json ../../outputs/drivelm_ds_smoke.json \
  --cache-file ../../outputs/deepseek_judge.sqlite \
  --judge-limit 12
```

Remove `--judge-limit` for a complete report. A full final score is emitted only
when reference/prediction coverage and semantic-judge coverage are both 100%.
