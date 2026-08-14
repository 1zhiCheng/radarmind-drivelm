# v0.40：DriveLM Planning Trajectory RL（GRPO vs GSPO）

## 版本结论

v0.40 从已晋级的 v0.39B MoL step 700 初始化 Planning 专家，将单条 QA
升级为带上游预测状态的 trajectory policy，在相同数据、rollout、训练预算下
对照 GRPO 与 GSPO。**GRPO-70 与 GSPO-90 均通过门槛，GSPO-90 正式晋级。**

## 模型与数据链路

```text
单帧六路同步相机
  -> 冻结 Perception / Prediction MoL 专家
  -> 预测 trajectory state（不含 gold 上游答案）
  -> Planning policy（GRPO 或 GSPO）
  -> 驾驶规划答案
  -> reward / 冻结 checkpoint selector / 最终语义评测
```

训练集包含 10,813 条 Planning QA，dev 包含 1,399 条，均为六路相机输入，
train/dev 场景重叠为零。上游上下文来自冻结专家预测，不直接输入 reference
Perception/Prediction 答案，避免标签泄漏并保持训练部署一致。

## 训练配置

| 配置项 | 数值 |
| --- | --- |
| Base VLM | Qwen2.5-VL-7B-Instruct |
| 初始化 | v0.39B `expert_planning/checkpoint-700` |
| 可训练参数 | Planning LoRA rank 8 / alpha 16 |
| GPU | 3 × RTX 5090（物理卡 0、2、3） |
| 并行与 rollout | FSDP2 + vLLM，4 rollouts/prompt |
| Train / val batch | 12 / 12 |
| Updates / LR | 100 / `5e-7` |
| Sampling | temperature 0.8，top-p 0.95 |
| Prompt / response limit | 3072 / 256 tokens |
| KL | low-var KL，0.01 |
| Save / validation | 每 10 step |
| 唯一算法变量 | GRPO `vanilla` vs GSPO `gspo` loss |

## 奖励与选择

```text
R = 0.40 * Token-F1 + 0.20 * ROUGE-L + 0.25 * Action-F1
  + 0.05 * Exact + 0.05 * GroundingValidity + 0.05 * FormatValidity
```

在线 reward 完全本地、确定性、无 API。DeepSeek-V4-Flash 只在 checkpoint
冻结后做最终语义判断，不参与训练或选点。selector 按完整 dev reward 选最高点，
同分再比较 Token-F1：

| 算法 | 选中 step | 内部 reward | 内部 Token-F1 |
| --- | ---: | ---: | ---: |
| GRPO | 70 | 0.606778 | 0.563953 |
| GSPO | 90 | **0.608048** | **0.565726** |

VERL 跑完 step 100 的 117 个 validation batch，但未打印 aggregate 行，因此严格
排除 step 100，只比较具有完整聚合值的 step 0–90。

## 最终同 ID 评测

三组模型均生成 1,399/1,399 条答案，DeepSeek judge 也均完成
1,399/1,399，没有 graph-gating 排除。

| 指标 | MoL baseline | GRPO-70 | **GSPO-90** |
| --- | ---: | ---: | ---: |
| Trajectory reward /100 | 60.8118 | 61.1674 | **61.2600** |
| Reward Token-F1 /100 | 55.8800 | 56.4231 | **56.5607** |
| Reward ROUGE-L /100 | 55.0403 | 55.5712 | **55.7057** |
| Action-F1 /100 | 66.3474 | **66.4189** | 66.3474 |
| Exact Match /100 | 17.2981 | 17.5840 | **18.1558** |
| DeepSeek Planning /100 | 71.2795 | 71.4582 | **71.5904** |

GSPO-90 相比 baseline，trajectory reward、Token-F1、ROUGE-L、Exact Match
和 Planning judge 分别提升 0.4482、0.6807、0.6654、0.8578、0.3109。
Action-F1 持平，说明提升并非简单堆叠动作词。

晋级要求：coverage 和 judge completion 均为 100%，reward/Token-F1 严格提升，
Planning judge 下降不超过 0.5。两者均通过；按 Planning judge、reward 排序，
最终选择 GSPO-90。

## 复现入口

- `run_v040_pipeline.sh`：数据、预检、smoke 和训练；
- `run_verl_rl.sh`：冻结的三卡配置；
- `trajectory_reward.py`：本地 reward；
- `summarize_training.py`：盲选 checkpoint；
- `run_final_eval.sh`：导出与同 ID 推理；
- `run_semantic_eval.sh`：语义评测与晋级。

紧凑结果位于 `final_eval/frozen_selection.json` 和
`final_eval/final_comparison.json`；本机 adapter 位于
`exports/grpo-step70/lora_adapter` 与 `exports/gspo-step90/lora_adapter`。

## 真实性边界与下一步

这是本地 scene-isolated Planning trajectory dev，不是官方隐藏榜单，也不代表
全任务 3,355 QA 同步提高。当前仍为单帧六路相机，没有混入雷达、LiDAR、
CARLA 或历史帧。后续以 GSPO-90 为唯一基线，比较更长 trajectory、多轮
credit assignment 或 Agentic rollout，并保留 camera-only control。
