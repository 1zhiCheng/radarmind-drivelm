# RadarMind v0.36：DriveLM 单帧六相机纯 CE 强基座实验

日期：2026-08-12
状态：B00 已训练并完成全量评测，不晋级；B10 权重准备中

## 1. 目标和边界

v0.36 只研究两个变量：VLM 基座容量和每路相机的视觉 token 预算。所有实验
只使用 DriveLM-nuScenes v1.1、单个当前关键帧和六路同步相机，不使用历史帧、
雷达、激光雷达、地图、真值框、ROI crop 或其他 oracle metadata。

SFT 固定使用 assistant-only 自回归交叉熵：

```text
L_CE = - sum_t log p(answer_t | six cameras, question, answer_<t)
```

图像、system、user、padding token 均以 `-100` mask；不使用 tag 权重、官方
指标权重、坐标专项权重、graph gating、DeepSeek 分数、DPO 或 RL reward。

## 2. 实验矩阵

| ID | 基座 | 每图最大视觉 token | 当前状态 |
| --- | --- | ---: | --- |
| B00 | Qwen2.5-VL-3B-Instruct | 128 | 完成，控制组，不晋级 |
| B10 | Qwen2.5-VL-7B-Instruct | 128 | 权重准备中 |
| B11 | Qwen2.5-VL-7B-Instruct | 256 | 可选，仅在 B10 诊断证明有必要时执行 |

B12 已取消。B00 不是为了替代 C00，而是为 B10/B11 建立相同 camera-only prompt、纯 CE
训练器和 DDP 路径下的容量控制组。

## 3. 新训练器

代码目录：

```text
/path/to/radarmind-drivelm/reproduction/qwen_vl_v036
```

训练器明确复用 `reproduction/qwen_vl/common.py` 的 camera-only collator，不再
使用旧 Stage-1 collator。旧 C00 数据虽然没有 history/ROI 图像，但训练时曾经
经过带 history/ROI 说明的 Stage-1 prompt；v0.36 从 prompt 中移除了这些残留，
只保留数据记录自身的六相机 system prompt 和原始问题。

64 条数据 dry-run 的单样本结果：

| 检查项 | 结果 |
| --- | ---: |
| visual inputs | 6 |
| 合并视觉 token（近似） | 720 |
| 总序列 token | 968 |
| assistant supervised token | 133 |
| masked token | 835 |

两张 RTX 5090 的 2-step DDP smoke 完成，LoRA/optimizer/processor 均可保存。

## 4. B00 正式训练

| 参数 | 值 |
| --- | --- |
| train QA | 26,095 |
| base model | Qwen2.5-VL-3B-Instruct |
| current cameras | 6 |
| max visual token/image | 128 |
| objective | pure assistant-token CE |
| LoRA | r=8, alpha=16, dropout=0.05 |
| learning rate | 2e-4 |
| max length | 4096 |
| effective global batch | 4 |
| optimizer updates | 6,524 |
| seed | 42 |

训练完成结果：

| 项目 | 结果 |
| --- | ---: |
| mean CE loss | 0.284627 |
| elapsed | 10,901.7 秒（约 3 小时 2 分钟） |
| seconds/update | 1.671 |
| trainable LoRA params | 18,576,384 |

模型产物：

```text
$DRIVELM_ROOT/models/qwen2.5-vl-3b-drivelm-v036-b00-seed42
```

目录内包含 adapter、processor、训练报告和带 optimizer state 的
`v036_training_state.pt`。

## 5. B00 全量 dev 结果

预测覆盖 3,355/3,355。以下均为固定 scene-isolated dev 的本地结果，不是官方
隐藏服务器分数。

### 5.1 全量 lexical 指标

| 模型 | Exact Match | Token-F1 | ROUGE-L | 多选题准确率 |
| --- | ---: | ---: | ---: | ---: |
| C00-CE | 42.325% | 72.294% | 70.150% | 82.360% |
| B00 | 42.593% | 72.753% | 70.635% | 81.798% |
| B00 - C00-CE | +0.268 pp | +0.459 pp | +0.485 pp | -0.562 pp |

2,000 次 paired bootstrap 显示总体 EM、Token-F1、ROUGE-L 的置信区间均跨 0，
因此不能把这些小幅变化解释为稳定提升。Planning 的 Token-F1 和 ROUGE-L
分别提升约 1.31 和 1.30 pp，置信区间略高于 0，但这没有转化为官方结构分数。

### 5.2 DriveLM-DS 官方结构代理指标

| 模型 | Eligible | Accuracy | Planning | Language | Coord F1 | Graph | Match | Final |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C00-CE | 1,919 | 78.366% | 69.234 | 47.738% | 15.404% | 48.220 | 31.812 | 0.592768 |
| v0.35 router | 1,919 | 80.787% | 69.234 | 47.738% | 15.404% | 48.220 | 31.812 | **0.597609** |
| B00 | 1,844 | 77.120% | 68.484 | 46.835% | 13.384% | 44.773 | 29.078 | 0.580001 |

B00 的完整语义评测需要 752 条，全部完成且无失败；其中 588 条命中冻结缓存，
164 条是新的 DeepSeek V4 Flash 调用。

## 6. 为什么 B00 不晋级

B00 的全量 lexical 指标略有提高，但 important-object 第一问的坐标/对象匹配
下降，使官方 graph gating 的 eligible 从 1,919 降到 1,844，多排除了 75 条
下游 QA。与此同时 Accuracy、语义 Planning、Language、坐标 F1、Graph 和
Match 都低于 C00-CE。因此 B00 未满足以下晋级条件：

- `Final >= v0.35 router + 0.003`；
- eligible 不低于 1,919；
- 四个官方分项没有明显退化。

这也说明仅比较 EM/F1/ROUGE-L 会得到错误结论；模型选择必须同时看公开
DriveLM 结构、gating 数量和固定共同 eligible 子集。

## 7. 结论和下一步

1. B00 作为 v0.36 camera-only 纯 CE 控制组永久保留，不替换 v0.35 router；
2. 不因为 B00 退化就修改损失函数，B10 仍保持完全相同的纯 CE，先单独检验
   3B 到 7B 的容量收益；
3. B10 负责完整跑通并审计 `权重校验 -> 数据审计 -> dry-run -> 2-step DDP
   smoke -> 全量训练 -> 断点产物校验 -> 3,355 条推理 -> lexical/DriveLM-DS
   评测 -> paired comparison -> 版本报告`；
4. B10 与 B00 使用同一数据 SHA-256、prompt、有效 batch、学习率、LoRA 和 seed；
5. B11 不是必跑项；只有 B10 完成后，若小目标/坐标 grounding 诊断明确指向
   视觉预算不足，才测试 256 token/image；B12 不再执行。

评测产物：

```text
$DRIVELM_ROOT/reproduction/qwen_vl_v036/b00_dev_predictions.json
$DRIVELM_ROOT/reproduction/qwen_vl_v036/b00_dev_metrics.json
$DRIVELM_ROOT/reproduction/qwen_vl_v036/b00_drivelm_ds.json
$DRIVELM_ROOT/reproduction/qwen_vl_v036/paired_c00ce_vs_b00.json
$DRIVELM_ROOT/reproduction/qwen_vl_v036/paired_c00ce_vs_b00.md
```
