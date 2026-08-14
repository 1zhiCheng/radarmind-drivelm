# RadarMind-DriveLM v0.39：自适应 Mixture of LoRA Experts

状态：完成；checkpoint-700 已通过全量与 same-ID DriveLM-DS，冻结为 trajectory RL 起点。

## 1. 为什么从单 LoRA 转向 MoL

DriveLM 的 Perception、Prediction、Planning 和 Behavior 输出分布差异明显。v0.37B 使用单个 rank-8 LoRA 同时拟合四类任务，容易出现不同任务之间的梯度干扰。v0.39 使用四个同构 LoRA 专家，按官方 QA hierarchy 中已有的任务键硬路由。路由只依赖问题元数据，不读取答案、dev reference 或 Judge 分数。

该方案不是稠密 MoE：Qwen2.5-VL-7B 主干保持共享，每条 QA 只激活一个 LoRA，因此推理时的活动参数量近似单 LoRA；代价是需要存储四份 adapter，并分别维护 checkpoint。

## 2. 公平对照

- `M00`：冻结 v0.37B checkpoint-75；
- `M01`：单一 shared LoRA，继续 100 次纯 CE 更新；
- `M10`：四任务 LoRA，每个继续 100 次纯 CE 更新；
- 相同 base、start adapter、rank 8、alpha 16、学习率 `2e-6`、seed 42、六相机输入和 128-token/image 视觉预算。

训练损失仍为 assistant-token-only autoregressive CE，没有把 dev 指标、DeepSeek 分数或 task weight 写入 loss。M10 的总训练计算约为 M01 的四倍，该差异在结果中显式披露。

v0.39A 的 100-step M10 已将全量 Final 从 v0.37B 的 0.596356 提升到 0.596955，并通过 1,799 条 same-ID 审计，证明 MoL 值得扩展。

## 3. 自适应训练控制器

v0.39B 从同一个 v0.37B 起点干净重训，第一轮保存 100/200/300/400/500。若最新点仍有实质提升，控制器按 200-step block 续训，并每 100 step 推理完整 3,355 QA。

主选择指标为全量 Token-F1，最小有效增益为 `0.0005`。同时要求：

- coverage = 100%；
- Planning Token-F1、MC accuracy 和 anchor coordinate F1 不越过退化阈值；
- graph eligible count 不出现异常减少；
- 连续两点没有有效增益则早停；
- 1500 step 为硬计算上限；
- DeepSeek 不参与 checkpoint sweep，只评测冻结候选。

因此 dev Judge 不会被反复查询来挑 checkpoint。训练异常退出可从 adapter、AdamW state 和 partial prediction JSONL 恢复。

## 4. 收敛曲线

| Step | Token-F1 | Planning Token-F1 | MC accuracy | Anchor F1 | Eligible |
| ---: | ---: | ---: | ---: | ---: | ---: |
| v0.39A | 73.2612% | 58.7023% | 84.2697% | 13.8777% | 1,874 |
| 100 | 73.1380% | 58.5225% | 84.1573% | 13.9646% | 1,859 |
| 200 | 73.3008% | 58.8414% | 84.1573% | 15.1236% | 1,917 |
| 300 | 73.7556% | 59.8071% | 84.2697% | 15.0829% | 1,892 |
| 400 | 73.8779% | 60.1389% | 84.1573% | 15.4389% | 1,917 |
| 500 | 74.2262% | 60.8854% | 84.2697% | 15.1785% | 1,914 |
| 600 | 74.3967% | 61.2724% | 84.2697% | 15.6346% | 1,906 |
| **700** | **74.5346%** | **61.4710%** | **84.4944%** | 15.3519% | 1,911 |
| 800 | 74.5074% | 61.3782% | 84.4944% | 15.4923% | 1,913 |
| 900 | 74.5086% | 61.2867% | 84.6067% | **16.1762%** | 1,938 |

800 和 900 的 Token-F1 均未比 700 提高 0.05 个百分点，patience=2 被触发。虽然 900 的 grounding 与 MC 更高，但预注册主指标没有改善，不能在观察结果后改规则，因此回滚到 700。

## 5. 最终评测

| 指标 | v0.39A | **v0.39B-700** | 差值 |
| --- | ---: | ---: | ---: |
| Exact Match | 43.6066% | **43.9940%** | +0.3875pp |
| Token-F1 | 73.2612% | **74.5346%** | +1.2735pp |
| ROUGE-L | 71.3344% | **72.4050%** | +1.0706pp |
| MC accuracy | 84.2697% | **84.4944%** | +0.2247pp |
| Planning Judge /100 | 70.7832 | **72.4769** | +1.6936 |
| Coordinate F1 | 13.3838% | **14.6465%** | +1.2626pp |
| Graph Judge /100 | 44.9242 | **46.8561** | +1.9318 |
| Match /100 | 29.1540 | **30.7513** | +1.5972 |
| DriveLM-DS Final | 0.596955 | **0.608245** | **+0.011290** |

graph gating 会改变可评测集合，因此还在两者共同的 1,812 个 eligible ID 上做配对审计。Final 从 0.593502 提升到 0.607388（+0.013886），Planning +2.488、MC +0.548pp、coordinate F1 +1.263pp，Judge 均完成 733/733，所有门槛通过。

这些仍是 scene-isolated local dev 和 DeepSeek proxy 结果，不是官方隐藏测试服务器成绩。

## 6. 产物与下一阶段

`best_checkpoint.json` 固定四个 checkpoint-700 adapter 路径和晋级证据。trajectory RL 不重新选择 SFT/MoL 起点，而是从该文件加载 policy。GRPO 与 GSPO 必须共享数据、reward、rollout 数、seed、更新次数和评测协议，只改变 policy ratio/loss aggregation，避免把算力或 reward 差异误记为算法提升。
