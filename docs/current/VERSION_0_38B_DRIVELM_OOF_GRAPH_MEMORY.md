# RadarMind-DriveLM v0.38B：OOF Graph Memory 与链式一致性 SFT

状态：oracle-memory feasibility probe 已完成；进入 train-only memory-consumption pilot，正式 OOF 训练尚未启动。

## 1. 设计依据

v0.38A 将 eligible QA 从 1,866 提高到 1,901，anchor coordinate F1 从 14.07% 提高到 14.83%，但 Final 从 0.596356 降到 0.592092。same-ID 审计显示共同 1,830 条 QA 的 Planning 分数完全相同；真正的问题是新放行的 35 条 Planning QA 平均只有 52.0 分。

因此下一阶段不再孤立优化“是否找到对象”，而是优化完整链路：第一问发现对象后，后续 perception、prediction 和 planning 必须消费同一个对象图并保持回答正确。

## 2. v0.38B-0：零训练路由诊断

诊断只让每帧第一个 graph anchor 使用 v0.38A checkpoint-50，其余 2,888 条 QA 全部使用冻结 v0.37B-75。它不改权重、不重新生成，也不根据 dev reference 选答案。

| 指标 | v0.37B-75 | v0.38A-50 | Anchor route |
| --- | ---: | ---: | ---: |
| Eligible QA | 1,866 | 1,901 | 1,901 |
| Anchor coordinate F1 | 14.0676% | 14.8332% | 14.8332% |
| Planning /100 | 70.8571 | 69.7295 | 69.7450 |
| Final | 0.596356 | 0.592092 | 0.592154 |

路由仅恢复 0.000062 Final，仍明显低于 v0.37B。这证明仅拼接“更好的 anchor”和“旧的下游回答”不能解决问题，必须显式训练 graph-memory 消费能力。

## 3. 严格 train-only OOF memory

按 scene 和 seed 42 做三折 cross-fitting。三个 anchor LoRA 各自只使用另外两折的 train anchor，在三张 RTX 5090 上并行训练，再对未见过的 held-out fold 生成第一问答案。拼接后，每个 train frame 恰好有一条 out-of-fold anchor prediction。

禁止使用以下捷径：

- 用已在全量 train 上训练的 v0.37B/v0.38A 伪造 OOF memory；
- 将 reference anchor 伪装成 predicted memory；
- 使用 dev/official-val 答案构造训练输入；
- 根据 dev Final 反复修改 fold 或样本。

第一问输出只经确定性 parser 转换：

```text
<GRAPH_MEMORY source="oof_prediction">
  <object_id, camera, x, y>
  ...
</GRAPH_MEMORY>
```

解析失败时写入 `<GRAPH_MEMORY status="invalid_or_empty"/>`，不能静默替换成 gold。下游 prompt 保留六路当前相机、原问题和 memory；传感器输入仍是单帧六相机。

## 4. G00/G10 控制实验

- `G00`：v0.38A-50 初始化，原始输入上的纯 CE replay control；
- `G10`：相同初始化、样本数、顺序、seed、视觉预算和 updates，只给下游 QA 增加 OOF graph memory。

两组均采用四任务均衡采样，同时保留 anchor、无 memory 原始 QA 和带 OOF memory 下游 QA。若需要少量 gold memory 做格式冷启动，其比例必须按预注册 schedule 衰减到 0，并独立报告；主结论只看 OOF-predicted memory。

## 5. 训练合同

本阶段属于新输入接口的 SFT，不做 DPO/GRPO：

```text
L = CE_target(answer | six_cameras, question, graph_memory)
```

- target：DriveLM train reference answer；
- 起始学习率 `5e-7`，纯 CE，最多 1 epoch；
- 3 × RTX 5090 DDP，G00/G10 有效 batch 完全一致；
- 保存 25/50/75/100% 四个进度点；
- 所有 checkpoint 完成 3,355 QA 无 API 评测后，只允许一个冻结候选进入 DeepSeek。

若 G10 不优于 G00，不能把变化归因于 memory；若两者都不如 v0.37B，整阶段不晋级。

## 6. 两阶段感知 Agent

每个 frame 只编码一次六路图像并维护帧内状态：先回答第一问，解析 object graph，写入当前帧 graph memory；后续 QA 读取 memory；parser 失败走显式 empty-memory fallback；帧结束立即清空，禁止跨帧或跨 scene 污染。最终提交仍输出标准 `id -> answer`。

## 7. 晋级门槛

必须报告完整 3,355 QA、same-ID common eligible、gated-out-as-zero、各 tag、两类 coordinate P/R/F1、parser failure rate 和额外延迟。相对 v0.37B-75，要求：

- coverage 和 Judge completion 为 100%；
- full Final 严格高于 0.596356；
- Planning 回退不超过 0.5 分；
- same-ID Final 不下降且 Planning 回退不超过 0.5；
- MC accuracy 不低于 84.1573%；
- eligible 不少于 1,866，期望不少于 1,901；
- anchor coordinate F1 严格高于 14.0676%；
- 新放行 Planning QA 平均 Judge 分不低于 65，且明显高于当前 52.0。

先做冻结的 oracle-memory 小样本上界诊断。若 oracle 也无增益，则停止本阶段；若 oracle 有效而 OOF 无效，优先修复 parser 和 exposure gap。MoL、GRPO、PPO、GSPO、SAPO 与 OPD 均不在本阶段混跑。

## 8. Oracle-memory feasibility probe 结果

probe 只选择 v0.38A 相对 v0.37B 新放行的 35 条 Planning QA。推理 prompt 使用同帧 reference anchor 的 3--6 个对象 tuple，因此该 JSONL 被明确标记为 dev-only，禁止训练和晋级。DeepSeek 评测只发送原始问题、reference 与候选答案，oracle memory 不进入 API payload。

| 指标 | 冻结 v0.37B | Oracle context | 差值 |
| --- | ---: | ---: | ---: |
| Changed answers | - | 2 / 35 | - |
| Exact Match | 8.57% | 8.57% | 0 |
| Token-F1 | 40.51% | 40.51% | 0 |
| ROUGE-L | 39.28% | 39.28% | 0 |
| Planning Judge /100 | 52.00 | 54.86 | +2.86 |

两组均完成 35/35 Judge、失败为 0，且全部命中已有缓存。部分协议不计算 Final。结果说明对象图携带少量有效信号，但冻结模型几乎忽略从未训练过的 memory schema，不能据此直接启动完整三折 OOF。

下一步先做 train-only pilot：按 scene 划分内部 train/validation，G00 与 G10 使用相同的纯 CE、样本数和 updates；G10 仅增加 gold/predicted graph-memory curriculum。只有 held-out-train 上 memory 使用率、Planning Token-F1 和动作一致性明显优于 G00，才投入正式三折 cross-fitting。
