# RadarMind-DriveLM v0.37B：任务均衡、CE 锚定的保守 DPO

状态：训练、四 checkpoint 全量 dev 推理与 DriveLM-DS 评测均已完成；**checkpoint-75 通过全部冻结门槛并晋级**。

## 1. 为什么做这一版

v0.37A 用 7,149 个偏好对做常规 DPO，MC accuracy 上升，但训练后期 planning、坐标 grounding 和 Final 明显下降。失败原因更符合偏好分布失衡与 DPO 过优化，而非单 LoRA 容量不足。因此 v0.37B 不更换模型、不增加 MoL，只控制三个变量：四任务均衡、较弱 DPO 更新和 chosen-answer CE 锚定。

## 2. 数据与泄漏控制

- 初始偏好池：v0.37A 的 7,149 个 train-only 高置信偏好对；
- 平衡后：4,104 对，perception、prediction、planning、behavior 各 1,026 对；
- 排列：四任务 round-robin；三卡、梯度累积 4 时，每次全局更新恰好包含每类 3 条；
- 与 3,355 条 scene-isolated dev 的 QA ID 交集为 0；
- 平衡文件 SHA-256：`1e61515d6e8d6852fb4b5bd13af319aaedd4c8c0f2bac767bcb38a2aa9cb8ae8`。

## 3. 模型与损失

策略和冻结参考策略均从 v0.36 B10（Qwen2.5-VL-7B、六路单帧相机、LoRA）初始化。对 chosen 回答 `y+` 和 rejected 回答 `y-`，使用：

```text
L_DPO = -log sigmoid(beta * ((log pi(y+|x) - log pi(y-|x))
                            - (log pi_ref(y+|x) - log pi_ref(y-|x))))
L = L_DPO + lambda * CE_chosen
```

其中 `beta=0.05`、`lambda=0.1`。CE 只计算 chosen assistant tokens，用来限制策略偏离原有语言建模能力。实现使用冻结 B10 的预计算 sequence log-prob，并以解析系数顺序反传，数学上保持完整 DPO 梯度，同时避免在一张卡上同时驻留两个 7B VLM。

## 4. 训练配置

| 项目 | 配置 |
| --- | --- |
| GPU | 3 × RTX 5090 |
| 全局步数 | 100 |
| 每卡 micro batch | 1 |
| 梯度累积 | 4 |
| 有效全局 batch | 12 |
| 学习率 | `1e-6` |
| 视觉预算 | 128 tokens/image，六路相机 |
| 最大序列长度 | 4096 |
| 保存点 | 25 / 50 / 75 / 100 |
| dropout | 关闭 |
| 训练时间 | 773.39 秒（12.89 分钟） |
| 最终均值 loss | 0.73423 |
| 偏好准确率 | 56.00% |

训练没有出现 NaN、OOM 或中断。短训练和较低偏好准确率是有意的：目标不是在训练偏好集上饱和，而是保住 B10 的通用能力。

## 5. 四 checkpoint 离线筛选

每个 checkpoint 都独立生成完整 3,355 条 dev 预测，coverage 均为 100%。选择规则在查看 DeepSeek 分数前冻结：总体 Token-F1 不低于 B10、planning Token-F1 不低于 B10、MC 必须提高；通过后按 Token-F1、planning Token-F1、MC 排序，只送一个候选做语义 judge。

| Checkpoint | EM | Token-F1 | ROUGE-L | Planning Token-F1 | MC | 进入 Judge |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 25 | 43.4277% | 72.9410% | 71.0069% | 58.2928% | 83.7079% | 否 |
| 50 | 43.5171% | 73.0888% | 71.1695% | 58.3838% | 84.1573% | 否 |
| **75** | **43.5768%** | **73.1231%** | **71.2034%** | **58.4370%** | **84.1573%** | **是** |
| 100 | 43.5469% | 73.0969% | 71.1697% | 58.4650% | 84.0449% | 是 |

checkpoint-75 的主排序指标更高，因此按预注册规则被选中；没有在看到 DeepSeek Final 后反向挑 checkpoint。

## 6. B10 与 v0.37B-75 最终结果

DriveLM-DS 保留公开 DriveLM 指标结构和 graph gating，仅把不可用的 GPT judge 替换成固定的 DeepSeek-V4-Flash 本地代理协议。它不是官方隐藏测试分数。v0.37B 需要的 762 个语义判断全部完成，失败为 0。

| 指标 | B10 | v0.37B-75 | 差值 |
| --- | ---: | ---: | ---: |
| Coverage | 100% | 100% | 0 |
| Exact Match | 43.4575% | 43.5768% | +0.1193pp |
| Token-F1 | 73.0000% | 73.1231% | +0.1232pp |
| ROUGE-L | 71.0756% | 71.2034% | +0.1278pp |
| MC accuracy | 83.8202% | 84.1573% | +0.3371pp |
| Planning /100 | 70.6348 | 70.8571 | +0.2223 |
| Coordinate F1 | 13.1313% | 13.3838% | +0.2525pp |
| Graph /100 | 43.9394 | 44.2803 | +0.3409 |
| Match /100 | 28.5354 | 28.8321 | +0.2967 |
| DriveLM-DS Final | 0.594636 | **0.596356** | **+0.001721** |

冻结门槛全部通过：Final 严格提升、planning 回退不超过 0.5、coordinate F1 回退不超过 0.5pp、MC 不低于 B10、coverage 和 judge complete 均为 100%。因此 v0.37B-75 成为当前本地最优 checkpoint。

## 7. 如何复现

完整命令见 [`reproduction/qwen_vl_v037b/README.md`](../../reproduction/qwen_vl_v037b/README.md)。关键顺序是：

1. 从 v0.37A train-only preference pool 构建严格四任务均衡数据；
2. 预计算并审计冻结 B10 reference log-prob；
3. 用三张同型号 RTX 5090 训练 100 步；
4. 对 25/50/75/100 全部运行 3,355-QA dev；
5. 用冻结的离线门槛选一个 checkpoint；
6. 仅对入选候选运行完整 DriveLM-DS。

模型权重不提交 Git。服务器上的晋级 adapter 位于：

```text
/mnt/data/zzy/drivelm/models/qwen2.5-vl-7b-drivelm-v037b-anchored-dpo-seed42/checkpoint-75
```

## 8. 结论与下一阶段

v0.37B 证明单 LoRA 在合适的数据配比和锚定目标下可以同时改善 lexical、MC、planning、graph 与 coordinate 指标，因此当前**没有证据支持引入 MoL**。本次提升幅度较小，应该表述为本地 dev 上的受控增益，而不是官方榜单结论。

下一阶段转向 v0.38 graph/coordinate grounding：保持 v0.37B-75 为冻结基线，只针对 important-object、坐标格式和 graph consistency 构建 train-only hard examples，并做独立消融。只有 v0.38 在完整 dev 和 DriveLM-DS 上继续过门槛，才考虑短程 GRPO；OPD 若需要把 train 文本或图像发往外部 teacher API，必须另行获得数据发送授权。
