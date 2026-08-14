# v0.40 trajectory RL: GRPO versus GSPO

The final promoted model is **GSPO checkpoint 90**. This stage turns the v0.39B
Planning expert into a trajectory policy and compares GRPO with GSPO under one
matched experimental budget.

## Experiment contract

- Initialization: Qwen2.5-VL-7B + v0.39B MoL Planning LoRA checkpoint 700.
- Input: one synchronized six-camera frame, question, and frozen
  Perception/Prediction predictions; no gold upstream answer is exposed.
- Data: 10,813 train and 1,399 scene-isolated Planning QA; zero scene overlap.
- Updated parameters: rank-8 Planning LoRA only.
- Compute: 3 x RTX 5090, FSDP2, vLLM, four rollouts per prompt.
- Budget: 100 updates, batch 12, LR `5e-7`, temperature `0.8`, top-p
  `0.95`, response limit 256, and KL coefficient `0.01`.
- Controlled variable: GRPO `vanilla` versus GSPO `gspo` policy loss.

The deterministic, API-free online reward is:

```text
R = 0.40 TokenF1 + 0.20 ROUGE-L + 0.25 ActionF1
  + 0.05 Exact + 0.05 GroundingValidity + 0.05 FormatValidity
```

DeepSeek-V4-Flash is used only after checkpoint selection as a semantic dev
judge; it is never used as a train reward.

## Final result

All models generated all 1,399 Planning answers and completed 1,399/1,399 judge
calls.

| Metric | MoL baseline | GRPO-70 | **GSPO-90** |
| --- | ---: | ---: | ---: |
| Trajectory reward | 60.8118 | 61.1674 | **61.2600** |
| Reward Token-F1 | 55.8800 | 56.4231 | **56.5607** |
| Reward ROUGE-L | 55.0403 | 55.5712 | **55.7057** |
| Action-F1 | 66.3474 | **66.4189** | 66.3474 |
| Exact Match | 17.2981 | 17.5840 | **18.1558** |
| DeepSeek Planning /100 | 71.2795 | 71.4582 | **71.5904** |

GSPO-90 improves trajectory reward by **+0.4482 points**, Exact Match by
**+0.8578 points**, and Planning judge by **+0.3109 points**. GRPO-70 also
passes every frozen gate, but GSPO-90 ranks higher by Planning judge and then
trajectory reward. These are local Planning results, not a hidden-server score
or a claim over all 3,355 task-family QA.

## Full-system completion

The selected RL Planning answers were routed back into the frozen v0.39B
Perception, Prediction, and Behavior experts. A raw 7B zero-shot control and
both RL candidates then completed all 3,355 QA and same-ID audits.

| Metric | Raw 7B | **MoL-700** | GRPO-70 | GSPO-90 |
| --- | ---: | ---: | ---: | ---: |
| Exact Match | 13.7109% | **43.9940%** | 43.5469% | 43.7854% |
| Token-F1 | 24.2340% | **74.5346%** | 74.0030% | 74.0246% |
| Planning /100 | 36.8308 | **72.4769** | 72.0910 | 72.0756 |
| DriveLM-DS Final | 0.257961 | **0.608245** | 0.606702 | 0.606640 |

All prediction and judge coverage is 100%. GRPO and GSPO use exactly the same
1,911 graph-eligible IDs as MoL-700, but both lower Final and fail the paired
promotion gate. **MoL-700 remains the full-system baseline.**

## Reproduce

```bash
cd /path/to/DriveLM-main
bash reproduction/qwen_vl_v040_trajectory_rl/run_v040_pipeline.sh
bash reproduction/qwen_vl_v040_trajectory_rl/run_final_eval.sh
bash reproduction/qwen_vl_v040_trajectory_rl/run_semantic_eval.sh
bash reproduction/qwen_vl_v040_trajectory_rl/run_full_system_completion.sh
```

The launchers expose their machine-specific paths at the top. Expected markers
are `TRAINING_COMPLETE`, `final_eval/OFFLINE_COMPLETE`, and
`final_eval/SEMANTIC_COMPLETE`.

Reliability checks include reward unit tests, six-image loader validation,
one-step smoke runs, validation/save every ten updates, greedy same-ID final
inference, and promotion gates for 100% coverage, 100% judge completion,
strict reward/Token-F1 gains, and Planning regression no worse than 0.5.

VERL completed every step-100 validation batch but omitted its aggregate line.
The frozen selector therefore uses only fully aggregated checkpoints 0–90.
