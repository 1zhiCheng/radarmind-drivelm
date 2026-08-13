# RadarMind v0.36：DriveLM 单帧六相机纯 CE 强基座实验

日期：2026-08-12
状态：B00、B10、B11 均已完成全量训练与评测；B10 保留，B11 不晋级

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
| B10 | Qwen2.5-VL-7B-Instruct | 128 | 已完成，v0.36 最优 |
| B11 | Qwen2.5-VL-7B-Instruct | 256 | 已完成，不晋级 |

B12 已取消。B00 不是为了替代 C00，而是为 B10/B11 建立相同 camera-only prompt、纯 CE
训练器和 DDP 路径下的容量控制组。

## 3. 新训练器

代码目录：

```text
/home/zhangzongyuan/Myproject/drivelm/DriveLM-main/reproduction/qwen_vl_v036
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
/mnt/data/zzy/drivelm/models/qwen2.5-vl-3b-drivelm-v036-b00-seed42
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

## 7. B00 后的既定执行计划（已完成）

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
/mnt/data/zzy/drivelm/reproduction/qwen_vl_v036/b00_dev_predictions.json
/mnt/data/zzy/drivelm/reproduction/qwen_vl_v036/b00_dev_metrics.json
/mnt/data/zzy/drivelm/reproduction/qwen_vl_v036/b00_drivelm_ds.json
/mnt/data/zzy/drivelm/reproduction/qwen_vl_v036/paired_c00ce_vs_b00.json
/mnt/data/zzy/drivelm/reproduction/qwen_vl_v036/paired_c00ce_vs_b00.md
```


## 8. B10/B11 最终验收（2026-08-13）

两组均使用相同的 26,095 条 QA、纯 assistant-token CE、有效全局 batch 4、
LoRA rank 8 和 seed 42。唯一受控变量为每路相机视觉预算。

| 项目 | B10：128 token/image | B11：256 token/image |
| --- | ---: | ---: |
| updates | 6,524 | 6,524 |
| mean train loss | 0.27292 | 0.27545 |
| 训练耗时 | 3.13 h | 5.50 h |
| dev coverage | 100% | 100% |
| Exact Match | 43.46% | 42.77% |
| Token-F1 | 73.00% | 72.75% |
| ROUGE-L | 71.08% | 70.93% |
| 多选准确率 | 83.82% | 82.02% |
| graph eligible | 1,889 | 1,779 |
| DriveLM-DS Accuracy | 79.91% | 75.55% |
| Planning (DeepSeek/100) | 70.63 | 70.68 |
| Language | 0.4760 | 0.4539 |
| coordinate F1 | 13.13% | 12.12% |
| Graph (DeepSeek/100) | 43.94 | 44.17 |
| Match (/100) | 28.54 | 28.14 |
| DriveLM-DS Final | **0.59464** | **0.58088** |

B10 judge 为 770/770，B11 judge 为 723/723，均为 100% 完成且无失败。
DriveLM-DS 使用 DeepSeek-V4-Flash 替代官方 GPT judge，是本地公开结构代理分数，
不是隐藏测试服务器成绩。

B11 的 Planning 与 Graph 语义项有极小提升，但 graph eligible 减少 110，
且 Accuracy、Language、coordinate F1、Match 和 Final 同时下降；其训练耗时增加
约 75.5%。因此更高视觉预算未带来总体收益，v0.36 选择 B10，B11 保留为
完整负结果，不进入下一阶段。

最终产物：

```text
/mnt/data/zzy/drivelm/models/qwen2.5-vl-7b-drivelm-v036-b10-seed42
/mnt/data/zzy/drivelm/models/qwen2.5-vl-7b-drivelm-v036-b11-seed42
/mnt/data/zzy/drivelm/reproduction/qwen_vl_v036/b10_dev_metrics.json
/mnt/data/zzy/drivelm/reproduction/qwen_vl_v036/b10_drivelm_ds.json
/mnt/data/zzy/drivelm/reproduction/qwen_vl_v036/b11_dev_metrics.json
/mnt/data/zzy/drivelm/reproduction/qwen_vl_v036/b11_drivelm_ds.json
/mnt/data/zzy/drivelm/reproduction/qwen_vl_v036/paired_b10_vs_b11.json
```
