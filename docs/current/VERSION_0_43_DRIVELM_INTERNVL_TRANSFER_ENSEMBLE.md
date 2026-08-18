# v0.43：从 InternVL4Drive 迁移的 Anchor Ensemble

## 1. 目标与结论

本版本研究 CVPR 2024 Driving with Language Outstanding Champion
`InternVL4Drive`，只迁移能够在现有 Qwen2.5-VL / MoL 流水线上做严格对照的
思想，不覆盖 MoL-700、Graph-A 或历史预测。

最终形成固定、无答案依赖的推理路由：

```text
每帧 qa_index=0 的 important-object anchor -> Graph-A
其余 Perception / Prediction / Planning / Behavior -> MoL-700
```

完整 3,355-QA DriveLM-DS Final 从 `0.608245` 提升到 **`0.612293`**；严格
same-ID 子集从 `0.609563` 提升到 **`0.610557`**，全部冻结门槛通过。因此
v0.43 是新的本地完整系统晋级者，但仍不是官方隐藏测试分数。

## 2. InternVL4Drive 做了什么

NJU-ImagineLab 的四页技术报告包含五个主要设计：

1. 以 InternVL-1.5（InternLM-20B + InternViT-6B）为基座，做全参数微调；
2. 将六路相机按 `FL/F/FR; BL/B/BR` 排成 2×3 全景图，并写入相机名；
3. 把 2688×896 全景图切成十二个 448×448 tile，再加一个全局 thumbnail；
4. 用 SAM 根据中心点生成 mask/bounding box，并把坐标归一化到 `[0,1000]`；
5. 训练 v1/v2 两个互补模型，并观察到 ensemble 可进一步提高分数。

其训练使用 64×A100、DeepSpeed ZeRO-3、全局 batch 1024、学习率 `2e-5`、
1 epoch。单模型 v2 在官方榜单得到 `0.6002`。报告同时给出一个重要负结果：
加入上一关键帧的 temporal 版本因数据格式问题下降到 `0.4600`。

来源：

- [NJU-ImagineLab technical report](https://opendrivelab.github.io/Challenge%202024/language_NJU-ImagineLab.pdf)
- [InternVL DriveLM domain-adaptation recipe](https://internvl.readthedocs.io/en/latest/internvl2.0/domain_adaptation.html)
- [OpenGVLab/InternVL](https://github.com/OpenGVLab/InternVL)

## 3. 哪些设计适合迁移

| InternVL 设计 | 当前 Qwen/MoL 状态 | 判断 |
| --- | --- | --- |
| 2×3 六视角全景图 | 当前用六张独立图并带 camera token | 值得做独立输入消融，但直接替换会产生训练/推理分布偏移 |
| 显式布局提示 | 已有相机名，尚未明确左右/前后拓扑 | 低风险，可作为下一轮同初始化 CE 对照 |
| Dynamic high resolution | B11 已验证视觉预算翻倍但 Final 下降 | 不应单独增加 token；需与布局联合验证 |
| 中心点转 bounding box | 当前只有中心点；Graph-A coordinate F1 已显著提升 | 有潜力，但必须处理 SAM 误框和输出反变换 |
| Temporal keyframe | 我们历史帧实验与其 temporal 实验都不稳定 | 暂不进入刷榜主线 |
| 互补模型 ensemble | MoL Planning 强、Graph-A anchor 强 | 可以立即用已有全量预测严格验证 |

本轮选择最后一项，因为它不需要新增标注、不改变训练集、不查看 dev reference
来选择单条答案，并能回答一个清晰问题：Graph-A 的 anchor 能否与 MoL 的下游
任务形成互补。

## 4. 方法

### 4.1 固定路由

对每条 dev record，只读取输入侧已有字段 `qa_index`：

```python
use_graph_a = int(record["qa_index"]) == 0
answer = graph_a[id] if use_graph_a else mol700[id]
```

路由与问题答案、reference、模型置信度和 dev 得分无关。3,355 条中：

- Graph-A anchor：467 条，每帧恰好一条；
- MoL-700 downstream：2,888 条；
- coverage：3,355/3,355。

Graph-A 首节点决定图中哪些 downstream QA 满足官方 coordinate gating；MoL
继续负责强项 Planning、Accuracy、Behavior 和 tag-3 输出。因此，这不是按样本
挑高分答案，而是一个可部署的任务/节点级 hard router。

### 4.2 对照组

- Baseline：v0.39B MoL-700；
- Candidate-A：仅 anchor 使用 Graph-A（晋级候选）；
- Candidate-B：anchor 与全部 tag-3 使用 Graph-A（探索候选）。

Candidate-B 的 132 个新 `candidate/reference/kind` 组合不在已有 Judge cache 中。
由于本轮没有获得对 v0.43 新 payload 的外发授权，严格 cache-only evaluator
拒绝生成不完整 Final；该候选不参与晋级。

### 4.3 评测协议

沿用公开 `challenge/evaluation.py` 的：

```text
Final = 0.4 * semantic_judge
      + 0.2 * language
      + 0.2 * match
      + 0.2 * accuracy
```

其中 Language 由 BLEU-1..4、ROUGE-L、CIDEr 合成；Match 由 coordinate F1 与
graph semantic judge 合成。本地语义裁判仍是 DeepSeek-V4-Flash 代理。

为保证本轮没有外发数据，新增 `evaluate_cache_only.py`：它不创建 OpenAI/API
client，只以 SQLite `mode=ro` 读取历史缓存，任意 miss 立即失败。Candidate-A
完成 791/791 cache hits。

## 5. 完整 3,355-QA 结果

| Metric | MoL-700 | v0.43 Anchor Ensemble | Delta |
| --- | ---: | ---: | ---: |
| Prediction coverage | 3,355/3,355 | 3,355/3,355 | 0 |
| Graph eligible | 1,911 | **1,927** | **+16** |
| Accuracy | 0.808735 | **0.813154** | **+0.004419** |
| Semantic judge /100 | 72.4769 | **73.0197** | **+0.5429** |
| BLEU-1 | 0.772320 | **0.789634** | +0.017315 |
| BLEU-2 | 0.709429 | **0.724770** | +0.015341 |
| BLEU-3 | 0.651638 | **0.665099** | +0.013461 |
| BLEU-4 | 0.595859 | **0.607520** | +0.011661 |
| ROUGE-L | 0.723957 | **0.724571** | +0.000613 |
| CIDEr | **0.200502** | 0.198875 | -0.001627 |
| Match /100 | 30.7513 | 30.7513 | 0 |
| DriveLM-DS Final | 0.608245 | **0.612293** | **+0.004048** |

Final 相对提升约 `0.67%`。Match 不变是预期结果：本候选只替换首个 tag-2
anchor，tag-3 grounding 仍由 MoL 输出；anchor 主要改变 graph eligibility 和
Language。

## 6. Same-ID 公平审计

MoL 与 v0.43 的 graph-eligible 交集为 1,848 条：

| Metric | MoL-700 | v0.43 | Delta |
| --- | ---: | ---: | ---: |
| Accuracy | 0.802528 | 0.802528 | 0 |
| Planning /100 | 73.1169 | 73.1169 | 0 |
| MC accuracy | 77.8966% | 77.8966% | 0 |
| Coordinate F1 | 14.6465% | 14.6465% | 0 |
| Match /100 | 30.7513 | 30.7513 | 0 |
| Language | 0.475440 | **0.480405** | **+0.004965** |
| Final | 0.609563 | **0.610557** | **+0.000993** |

冻结门槛全部通过：相同数量、Judge 完整、Final 严格提高、Planning 回退不超过
0.5、coordinate F1 回退不超过 0.5pp、MC 不下降。

需要同时披露一个负向观察：same-ID lexical Token-F1 `-0.000969`、本地
ROUGE-L-F1 `-0.000474`，而官方 COCO Language 组合提高。这来自指标定义不同，
不能隐藏，也说明后续仍应进行官方隐藏集验证。

## 7. 与 InternVL4Drive 的关系

本版本不是 InternVL-1.5 的参数级复现，也不能把两边数值直接比较：

- InternVL 报告是官方 leaderboard GPT score；本项目是本地 scene-isolated dev
  和 DeepSeek proxy；
- InternVL 使用 64×A100 全参数微调；本项目复用 7B LoRA/MoL/Graph 产物；
- 本轮验证的是其“互补模型 ensemble”思想，不是声称复现 `0.6002`。

真正可借鉴的研究结论是：DriveLM 的 graph gating 使 anchor perception 成为独立
系统瓶颈。将“anchor 专家”和“下游推理专家”解耦，比让同一个 Adapter 同时兼顾
坐标召回与 Planning 更稳定。

## 8. 后续实验

1. `L00/L10`：保持六张独立图，仅加入显式物理布局 prompt，做同初始化 CE 对照；
2. `M00/M10`：六图输入对比 2×3 mosaic + global thumbnail，保持总 visual token
   预算一致；
3. `B00/B10`：用 SAM/检测器生成 train-only bounding-box 辅助监督，但提交输出仍
   反变换为官方中心点；
4. 将 v0.43 固定路由导出官方 val JSON，提交隐藏服务器，避免仅依赖代理 Judge；
5. 若继续 Graph 训练，使用 predicted-context/DAgger，避免 v0.42 的 teacher
   forcing exposure mismatch。

## 9. 复现产物

```text
reproduction/qwen_vl_v043_internvl_transfer/
  build_fixed_ensemble.py
  test_fixed_ensemble.py

/mnt/data/zzy/drivelm/reproduction/qwen_vl_v043_internvl_transfer/
  anchor/build_report.json
  anchor/ensemble_predictions.json
  anchor/drivelm_ds.json
  anchor_tag3/...
  same_id_vs_mol700/...
```

当前晋级关系更新为：

```text
MoL-700 (0.608245)
  -> InternVL-inspired fixed anchor ensemble (0.612293) 🏆
```
