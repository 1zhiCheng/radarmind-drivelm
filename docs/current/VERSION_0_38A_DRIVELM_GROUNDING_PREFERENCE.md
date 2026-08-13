# RadarMind-DriveLM v0.38A：Graph Anchor 与坐标 Grounding 偏好优化

状态：已完成；checkpoint-50 通过离线筛选，但未通过最终晋级门槛，v0.37B-75 继续作为主模型。

## 1. 起点与目标

本阶段从已晋级的 v0.37B checkpoint-75 开始，不更换 Qwen2.5-VL-7B、不改变六路相机输入和 128-token/image 视觉预算。目标不是泛化地“再做一轮 DPO”，而是同时解决：

1. 每帧第一条 important-object 回答的坐标错误会通过公开 graph gating 排除下游 QA；
2. tag-3 notice-graph 回答的对象坐标 F1 仍只有 13.38%；
3. 后训练不能用减少 eligible QA 的方式换取更高条件均值。

## 2. v0.37B 后新增的公平性约束

v0.37B 的全协议 eligible QA 为 1,866，B10 为 1,889。两模型在严格共同的 1,807 个 eligible ID 上重新评测后，v0.37B Final 仍提高 0.001056，说明晋级成立。自 v0.38 起，每个候选必须同时报告：

- 3,355 条完整 dev 的公开式 gated 指标；
- 相对冻结基线的 same-ID common-eligible 指标；
- eligible QA 总数及 tag 0/1/2/3 分布；
- important-object 与 tag-3 的坐标 precision/recall/F1；
- gated-out-as-zero 敏感性结果。

## 3. 为什么废弃旧截断候选

对 v0.37A 的 B10 train-only important-object 候选审计得到：

- 3,605 条第一问，共 7,210 个 sampled answers；
- 4,042 个候选被 128-token 上限截断；
- 7,205 个存在官方式坐标漏检，7,181 个存在坐标误检；
- 6,586 个会减少至少一条下游 eligible QA。

截断候选适合作为格式错误诊断，但大量用于偏好训练会让模型只学会“回答更长”，不能证明视觉 grounding 变好。因此 v0.38A 从当前 v0.37B-75 重新生成 256-token 候选，并强制排除不完整结构和极端长度差。

## 4. 数据构建

数据只来自 scene-isolated train，不使用 dev/official-val 参考答案，也不调用外部 API。

### 4.1 Fresh grounding candidates

- important-object anchor：3,605 条 `qa_index=0, tag=2`；
- notice graph：1,206 条 `tag=3`，其中 qa_index 1/2 为 171/1,035；
- 每条用 v0.37B-75 采样 2 个候选；
- `max_new_tokens=256, temperature=0.8, top_p=0.9`；
- 三张 RTX 5090 生成 anchor，A6000 并行生成 tag-3。

候选只有满足以下条件才可成为 rejected：

- `<object_id,camera,x,y>` tuple 完整闭合；
- 官方坐标 parser 的 pair 数与完整 tuple 数一致；
- 输出不是截断文本；
- rejected/chosen token 长度比在 `[0.60, 1.45]`；
- coordinate F1 不高于 0.75，或第一问至少导致一条下游 QA gating loss。

同一 QA 只保留一个综合 `gating loss + coordinate FP/FN/F1` 最难候选。

### 4.2 四任务均衡 replay

总数据仍为每任务 1,026 对、共 4,104 对，round-robin 排列：

- perception：优先使用 fresh important-object grounding pairs；
- prediction：优先使用 fresh tag-3 grounding pairs，不足部分由 v0.37B replay 补齐；
- planning、behavior：保持 v0.37B 的审计后 replay pairs；
- 所有 pair 的 reference log-prob 都用冻结 v0.37B-75 重新计算，禁止沿用 B10 分数。

这使 v0.38A 相对 v0.37B 的主要变量是 grounding pair 质量，而不是任务比例、模型大小或训练长度。

## 5. 训练合同

如果 fresh 数据审计满足数量门槛，则沿用 v0.37B 的保守目标：

```text
L = DPO(beta=0.05) + 0.1 * CE_chosen
```

- 初始化与 reference：v0.37B checkpoint-75；
- 3 × RTX 5090；有效全局 batch 12；
- learning rate `1e-6`，最多 100 steps；
- 保存 25/50/75/100；
- 所有 checkpoint 完整生成 3,355 QA 后再选择。

在候选完整率、fresh pair 数或 smoke gradient 未通过前不得启动正式训练。

## 6. 晋级门槛

相对 v0.37B-75，候选必须同时满足：

- coverage 与 judge completion 均为 100%；
- full-protocol Final 严格提高；
- eligible QA 不少于 1,866，并以恢复到 B10 的 1,889 为期望；
- coordinate F1 严格提高；
- full-protocol planning 回退不超过 0.5；
- same-ID common-subset Final 不下降，planning 回退不超过 0.5；
- MC accuracy 不低于 v0.37B；
- 不能仅靠更高输出长度或格式有效率解释收益。

任何一项失败都保留 v0.37B-75。

## 7. MoL、OOF memory 与 RL

- 本版不做 MoL：v0.37B 已证明单 LoRA 能让多个任务同步提升，尚无专家容量证据；
- 本版先修复 anchor/coordinate 监督质量，随后才做 v0.38B OOF graph memory，避免把错误第一问作为 agent memory 传播；
- 本版不做 GRPO/PPO/GSPO/SAPO；reward 与完整 DriveLM-DS 的相关性尚未校准；
- OPD 若需发送 train 文本或图像给外部 teacher，需要单独获得用户的数据发送授权。

## 8. 实际训练与评测结果

三张 RTX 5090 完成 100 steps，耗时 778.7 秒；训练集为 4,104 个 train-only 偏好对，reference log-prob coverage 为 100%。step 25/50/75/100 均完成 3,355 条 dev 推理。冻结的无 Judge 选择器只允许 step 50 进入 DeepSeek 评测。

| 指标 | v0.37B-75 | v0.38A-50 | 差值 |
| --- | ---: | ---: | ---: |
| Coverage | 100% | 100% | 0 |
| Exact Match | 43.5768% | 43.6066% | +0.0298 pp |
| Token-F1 | 73.1231% | 73.1039% | -0.0192 pp |
| MC accuracy | 84.1573% | 84.1573% | 0 |
| Eligible QA | 1,866 | 1,901 | +35 |
| Anchor coordinate F1 | 14.0676% | 14.8332% | +0.7656 pp |
| Planning /100 | 70.8571 | 69.7295 | -1.1276 |
| DriveLM-DS Final | 0.596356 | 0.592092 | -0.004264 |

在两模型共同 eligible 的 1,830 条 QA 上，Planning 完全不变，Final 仅因 Language 下降而从 0.594798 降至 0.594246。v0.38A 新放行 71 条、丢失 36 条；其中新增 35 条 Planning QA 的推导平均 Judge 分只有 52.0，而共同子集为 70.76，被替换掉的 18 条为 74.17。因此 full-protocol Planning 下跌主要是 graph gating 恢复了更难的下游题，而不是共同题上的规划能力被普遍破坏。

所有 judge 项完成且无失败，但 Final、Planning guardrail 和 same-ID Final 三项未通过。v0.38A 不晋级，也不继续追加 DPO；其 checkpoint-50 只保留为 grounding/anchor 实验组件。
