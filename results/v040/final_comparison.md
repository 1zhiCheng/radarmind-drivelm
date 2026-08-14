# v0.40 trajectory RL final comparison

> Local 1,399-row scene-isolated planning trajectory dev. DeepSeek-V4-Flash replaces the public GPT judge; this is not a hidden-server score.

| Metric | Baseline | GRPO-70 | GSPO-90 |
| --- | ---: | ---: | ---: |
| Trajectory reward | 60.8118 | 61.1674 | 61.2600 |
| Token-F1 | 55.8800 | 56.4231 | 56.5607 |
| ROUGE-L | 55.0403 | 55.5712 | 55.7057 |
| Action-F1 | 66.3474 | 66.4189 | 66.3474 |
| Exact Match | 17.2981 | 17.5840 | 18.1558 |
| Planning judge /100 | 71.2795 | 71.4582 | 71.5904 |

Promoted model: **gspo90**

## Frozen promotion gates

### grpo70

- [x] coverage_100_percent
- [x] judge_complete
- [x] trajectory_reward_strictly_improved
- [x] token_f1_strictly_improved
- [x] planning_deepseek_regression_at_most_0_5
### gspo90

- [x] coverage_100_percent
- [x] judge_complete
- [x] trajectory_reward_strictly_improved
- [x] token_f1_strictly_improved
- [x] planning_deepseek_regression_at_most_0_5
