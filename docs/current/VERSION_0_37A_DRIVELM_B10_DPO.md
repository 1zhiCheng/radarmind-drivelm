# RadarMind-DriveLM v0.37A: B10 离线 DPO

状态：全量候选、偏好构建、reference、三卡 DPO、五点 checkpoint sweep 和完整 dev 评测均已完成；DPO 不晋级，B10 继续作为主线。

## 1. 实验目的

v0.37A 检验一个单变量问题：在 v0.36 最优 B10 上，使用模型自产候选构建离线偏好对并进行标准 DPO，能否提升 DriveLM 指标。

固定不变：

- Qwen2.5-VL-7B-Instruct 基座；
- 单帧六路相机和每图 128 visual tokens；
- 26,095 条 train QA 与 3,355 条 scene-isolated dev QA；
- B10 的 LoRA 结构、prompt 和 dev 评测协议；
- dev reference 不参与候选生成或训练。

## 2. 数据流水线

三张 RTX 5090 将 26,095 条 train QA 确定性分为三个互斥分片，每题从冻结 B10 采样两个候选。batch 16 下完成全量生成，三个分片分别为 8,699、8,698、8,698 条。

偏好筛选只使用确定性高置信规则，不调用外部语义 API：

| 项目 | 数值 |
| --- | ---: |
| 候选覆盖 | 26,095 / 26,095 |
| 高置信偏好对 | 7,149 |
| Perception | 3,791 |
| Prediction | 1,254 |
| Planning | 1,078 |
| Behavior | 1,026 |
| train/dev scene overlap | 0 |
| dev ID 泄漏 | 0 |

chosen 是 train reference，rejected 是 B10 采样错误答案。模糊自由文本直接丢弃，train 文本没有发送给 DeepSeek。

## 3. 训练配置

冻结 B10 对每个 chosen/rejected 预计算 reference log-prob，三卡合并覆盖为 7,149/7,149。正式 DPO 从 B10 初始化：

| 配置 | 数值 |
| --- | --- |
| GPU | 3 x RTX 5090 |
| effective global batch | 12 |
| steps | 596 |
| beta | 0.1 |
| learning rate | 5e-6 |
| precision | BF16 |
| visual budget | 128 tokens/image |
| final mean DPO loss | 0.43876 |
| final preference accuracy | 86.91% |
| train time | 4,672 s |

reference 和 policy 使用相同 BF16 autocast 路径。训练前真实样本验收满足 policy margin 等于 reference margin、DPO logit 为 0、loss 为 ln(2)=0.693147。

## 4. 最终结果

所有结果均来自同一 3,355-QA 本地 scene-isolated dev，不是官方隐藏服务器分数。

| Metric | B10 | DPO step 596 | Delta |
| --- | ---: | ---: | ---: |
| Coverage | 100.00% | 100.00% | 0 |
| Exact Match | 43.46% | 42.98% | -0.48 pp |
| Token-F1 | 73.00% | 71.76% | -1.24 pp |
| ROUGE-L | 71.08% | 69.40% | -1.68 pp |
| MC accuracy | 83.82% | 84.83% | +1.01 pp |
| Planning /100 | 70.63 | 66.52 | -4.11 |
| Coordinate F1 | 13.13% | 6.31% | -6.82 pp |
| DriveLM-DS Final | 0.59464 | 0.55330 | -0.04134 |

DeepSeek judge 完成 750/750，失败为 0。最终 checkpoint 只提高多选准确率，但显著损害 planning、语言与坐标 grounding，因此不晋级。

## 5. Checkpoint sweep

五个中间 checkpoint 均完成 3,355/3,355 离线推理。离线指标显示 100 至 300 step 是合理早停窗口，400 step 后退化加速。

| Variant | EM | Token-F1 | ROUGE-L | MC accuracy | Planning Token-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| B10 | 43.46% | 73.00% | 71.08% | 83.82% | 58.43% |
| step 100 | 43.55% | 73.17% | 71.22% | 84.27% | 58.43% |
| step 200 | 43.67% | 73.13% | 71.14% | 84.27% | 58.33% |
| step 300 | 43.49% | 73.12% | 71.02% | 83.93% | 58.53% |
| step 400 | 43.31% | 72.85% | 70.66% | 84.04% | 58.19% |
| step 500 | 43.58% | 72.69% | 70.50% | 84.49% | 58.25% |
| step 596 | 42.98% | 71.76% | 69.40% | 84.83% | 56.41% |

合理候选进一步完成 DriveLM-DS：

| Variant | Planning /100 | Final | 晋级 |
| --- | ---: | ---: | --- |
| B10 | 70.63 | 0.59464 | 保留 |
| step 100 | 69.75 | 0.59430 | 否 |
| step 200 | 69.72 | 0.59196 | 否 |
| step 300 | 66.42 | 0.57561 | 否 |

step 100 仅比 B10 低 0.00034，但严格门槛要求 Final 提升且 planning 不显著回退，因此仍不替换 B10。

## 6. 失败分析与下一步

偏好对中 perception 占 53.0%，grounding_low_similarity 占 67.3%。纯 DPO 会持续扩大 chosen/rejected margin，却没有显式保持原始 chosen 的语言建模能力；训练越久，多选和 behavior 越好，而 planning、开放文本和精确坐标越差。

v0.37B 从原始 B10 重新初始化，不继承任何 v0.37A DPO 权重：

1. 四任务均衡采样，避免 perception 偏好主导；
2. 降低 learning rate 和 beta；
3. 在 DPO loss 上增加 chosen-answer CE 锚定；
4. 采用 25/50/75/100 step 高频早停；
5. 先做离线 sweep，仅对最优候选调用完整语义 judge。

## 7. 为什么暂不使用 MoL

Mix of LoRA 适合已经确认存在稳定任务专家和可学习路由信号的场景。目前最主要问题是偏好分布失衡与纯 DPO 过优化，而不是单 LoRA 容量不足。现在引入 MoL 会同时改变专家数量、路由、训练数据和优化目标，破坏可归因性。

MoL 保留为后续独立消融：只有 v0.37B 的单 LoRA 保守 DPO 稳定后，且不同任务的最佳 checkpoint 或梯度冲突仍明显，才比较单 LoRA 与多专家 LoRA。
