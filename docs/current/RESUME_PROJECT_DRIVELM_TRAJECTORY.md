# RadarMind-DriveLM：自动驾驶多视角感知与决策 VLM Agent

## 简历可直接使用版本

**RadarMind-DriveLM｜自动驾驶多视角感知与决策 VLM Agent｜独立研发**

技术栈：PyTorch、Transformers、Qwen2.5-VL-7B、LoRA/PEFT、DDP/FSDP、vLLM、verl、DPO、GRPO/GSPO、DriveLM-nuScenes、DeepSeek API

- 基于 DriveLM-nuScenes 构建单帧六路相机 Graph-VQA 流水线，完成 26,095 条训练 QA 与 3,355 条 scene-isolated dev QA 的数据校验、SFT、分布式推理和评测，覆盖 Perception、Prediction、Planning、Behavior 四级自动驾驶推理任务。
- 以 Qwen2.5-VL-7B 为骨干进行 assistant-token-only CE-LoRA 微调，通过 B00/B10/B11 控制变量实验确定 128 visual tokens/image 的 7B 基线；设计 3×RTX 5090 离线 DPO 流水线，完成候选生成、7,149 组偏好对构建、冻结 reference log-prob 预计算及多 checkpoint 晋级审计。
- 针对多任务梯度干扰实现四专家 Mixture of LoRA，按答案无关的官方 task key 硬路由，并设计每 100 step 自动评测、patience early-stop 与同 ID 公平性审计；最优 step 700 将本地 DriveLM-DS 由 0.59464 提升至 0.60824（相对 +2.29%），Token-F1 达 74.53%，多选准确率达 84.49%。
- 复现公开 DriveLM-DS 评测结构，集成 Accuracy、Planning/Graph 语义评判、语言质量、目标坐标 grounding 与 Match Score；使用带缓存和完整性校验的 DeepSeek-V4-Flash 替代不可用 GPT judge，并设置 coverage、judge completeness、planning 与 same-ID promotion gates，避免仅凭单项指标错误晋级。
- 面向 Agentic RL 搭建 trajectory 训练链路：冻结 Perception/Prediction 专家生成无 gold 泄漏的轨迹状态，仅优化 Planning policy；实现 API-free 组合奖励、四路 rollout、KL 约束以及三卡 GRPO/GSPO 对照。GSPO-90 在 1,399 条 Planning dev 上将 trajectory reward、Exact Match、DeepSeek Planning 分别提升 0.448、0.858、0.311 个百分点，并以 100% coverage/judge completeness 晋级。

项目地址：<https://github.com/1zhiCheng/radarmind-drivelm>

## 一页简历压缩版

如果版面只能容纳 3 条，使用下面版本：

- 基于 DriveLM-nuScenes 与 Qwen2.5-VL-7B 搭建六路相机自动驾驶 Graph-VQA Agent，完成 26,095 train / 3,355 scene-isolated dev QA 的 CE-LoRA 训练、三卡推理与 Perception→Prediction→Planning→Behavior 分层评测。
- 实现候选采样、冻结 reference 打分、DPO 和四专家 Mixture of LoRA；通过自适应 checkpoint sweep 与同 ID 晋级门控，将本地 DriveLM-DS 从 0.59464 提升至 0.60824（相对 +2.29%），Token-F1 74.53%、多选准确率 84.49%。
- 搭建无标签泄漏的 trajectory RL 流水线，以上游专家预测构造决策状态，使用可解释组合 reward、vLLM rollout、KL 约束及 verl/FSDP 在 3×RTX 5090 上对照 GRPO/GSPO；GSPO-90 在 1,399 条 Planning dev 上取得 61.260 trajectory reward 和 71.590 DeepSeek Planning，优于 MoL 基线并通过自动晋级门控。

## 面试时的 60 秒介绍

这个项目不是普通的图像问答微调，而是把 DriveLM 的六路环视图像组织成“感知、预测、规划、行为”四级自动驾驶推理 Agent。我先用严格控制变量的 CE-LoRA 得到 Qwen2.5-VL-7B 基线，再完整验证 DPO；实验发现标准 DPO 虽提高多选题，却会损害开放式 planning，因此增加 CE anchor、全量 dev 和晋级门控。随后，我将四类任务拆成独立 LoRA 专家并按问题任务路由，把本地 DriveLM-DS 从 0.59464 提升到 0.60824。最后构建无 gold 泄漏的 trajectory context，在相同预算下对照 GRPO 与 GSPO；GSPO-90 在完整 Planning dev 的确定性与语义指标上均优于 MoL 基线，并通过冻结晋级门控。

## 项目架构

```text
DriveLM-nuScenes 六路相机
          |
          v
数据审计与 scene-isolated split
          |
          v
Qwen2.5-VL-7B + CE-LoRA 基线
          |
          +----> 离线候选生成 -> DPO/CE-anchor 消融
          |
          v
Perception / Prediction / Planning / Behavior 四专家 LoRA
          |
          +----> 自适应 checkpoint sweep -> DriveLM-DS -> same-ID gate
          |
          v
冻结上游专家生成 trajectory state（无 gold 泄漏）
          |
          v
Planning policy：GRPO vs GSPO -> 1,399 Planning QA 完整 dev 评测
```

## 指标口径与真实性边界

- `0.60824` 是本地 scene-isolated dev 上的 DriveLM-DS 公开结构代理分数，不是官方隐藏 challenge server 分数。
- DeepSeek 仅用于最终 dev 语义评测，不作为 trajectory 在线训练 reward。
- trajectory 数据构建、奖励函数、三卡训练和自动验收流水线已经落地；GRPO/GSPO 均完成 1,399 条同 ID Planning dev 评测，GSPO-90 已按冻结规则晋级。该提升仅代表 Planning 子集，不写成 3,355 条全任务或官方隐藏榜单提升。
- CARLA 闭环控制属于独立的 `radarmind-carla` 项目，不混入本项目的离线 DriveLM 指标。

## ATS 关键词

多模态大模型、视觉语言模型、自动驾驶 Agent、VLM、Qwen2.5-VL、LoRA、PEFT、SFT、DPO、Mixture of LoRA、GRPO、GSPO、Agentic RL、Trajectory、Reward Design、FSDP、DDP、vLLM、verl、nuScenes、DriveLM、多视角视觉、Graph-VQA、模型评测、分布式训练
