# RadarMind v0.31：DriveLM-nuScenes 六视角 Qwen-VL 复现

## 1. 版本定位

v0.31 是 RadarMind 双域路线中的真实道路视觉基线：在 DriveLM-nuScenes v1.1 官方基础 QA 协议上，以 Qwen2.5-VL-3B-Instruct 和 LoRA 完成六相机 SFT、scene-isolated dev 推理、离线评估以及官方无标签 val 输出。

它对应 v0.32 文档中的 `historical A0`。v0.31 不包含雷达输入、CARLA 控制或时序历史帧，也不等同于论文中未完整公开训练实现的 BLIP-2 DriveLM-Agent。项目保留官方的数据抽取规则、问题 ID、六相机输入和提交格式，用可本地训练的开源 VLM 实现现代等价复现。

## 2. Pipeline

```text
DriveLM-nuScenes v1.1 annotations + six synchronized cameras
  -> official basic-QA extraction
  -> deterministic scene-level train/dev split
  -> Qwen2.5-VL-3B LoRA SFT
  -> 3,355 labeled-dev predictions
  -> EM / Token-F1 / ROUGE-L / MC / task metrics
  -> 15,480 official-val answers
```

## 3. 数据与隔离划分

| split | scenes | QA | label | purpose |
| --- | ---: | ---: | --- | --- |
| train | 619 | 26,095 | 有 | LoRA SFT |
| local dev | 77 | 3,355 | 有 | 本地离线评估 |
| official val | 149 | 15,480 | 无 | 挑战提交格式生成 |

构建器固定 `seed=42`。train/dev 按 scene 切分，同一 scene 不会跨集合泄漏；29,226 张唯一相机图像检查结果为缺失 0。每条样本固定使用当前时刻六路相机，顺序为 front、front-left、front-right、back、back-left、back-right。

数据目录：

```text
$DRIVELM_ROOT/data/QA_dataset_nus/
$DRIVELM_ROOT/data/nuscenes/samples/CAM_*/
$DRIVELM_ROOT/reproduction/qwen_vl/
```

生成命令：

```bash
cd /path/to/radarmind-drivelm
python reproduction/qwen_vl/build_dataset.py \
  --train-json $DRIVELM_ROOT/data/QA_dataset_nus/v1_1_train_nus.json \
  --val-json $DRIVELM_ROOT/data/QA_dataset_nus/v1_1_val_nus_q_only.json \
  --output-dir $DRIVELM_ROOT/reproduction/qwen_vl \
  --seed 42 --dev-ratio 0.1
```

## 4. 模型与训练配置

| item | value |
| --- | --- |
| base model | Qwen2.5-VL-3B-Instruct |
| LoRA trainable params | 18,576,384（0.4923%） |
| LoRA rank / alpha | 8 / 16 |
| batch size | 4 |
| learning rate | 2e-4 |
| visual budget | 75,264 pixels/image，约 `96×28×28` |
| total optimizer updates | 6,524 |
| final continuation mean loss | 0.341855 |

最终模型：

```text
$DRIVELM_ROOT/models/qwen2.5-vl-3b-drivelm-sixcam
```

训练在 step 3,000 后从 LoRA checkpoint 恢复并继续 3,524 步。模型参数连续，但旧训练器没有保存 Adam optimizer state，因此恢复段重新初始化了优化器动量。这个限制在 v0.32 中通过另训 uninterrupted A0-control 显式处理，不能隐藏。

## 5. Local-dev 正式结果

3,355 条预测全部匹配 reference，coverage 为 100%。以下是本地 scene-isolated dev 指标，不是官方隐藏服务器分数。

| task | n | Exact Match | Token-F1 | ROUGE-L |
| --- | ---: | ---: | ---: | ---: |
| Overall | 3,355 | **44.26%** | **73.33%** | **71.59%** |
| Perception | 890 | 45.51% | 82.56% | 78.56% |
| Prediction | 599 | 75.13% | 90.41% | 89.05% |
| Planning | 1,399 | 22.16% | 61.76% | 60.72% |
| Behavior | 467 | 68.52% | 68.52% | 68.52% |

多选题 890 条，准确率为 **81.46%**。对象标识简化结构匹配 precision/recall/F1 为 30.98%/32.82%/31.87%。主要短板是 planning、important-object、notice-graph 和精确对象坐标 grounding。

## 6. 官方 val 生成

无公开答案的 official val 已生成 15,480/15,480 条，唯一 ID 15,480、空答案 0，schema 全部为 `id + answer`。其中 1,543 条多选回答规范化为单个 A/B/C/D 字母。该文件只能上传挑战服务器获得官方指标。

```text
$DRIVELM_ROOT/reproduction/qwen_vl/official_val_output.json
SHA-256 77bdaca3d8c777020b9f043ce607557e5275aef7929048f49ee15a8f17cb2185
```

## 7. 可审计资产

| artifact | SHA-256 |
| --- | --- |
| final LoRA adapter | `1473386e2f1d45b1d269c60216d064ebfff42a62f7d3418bd40cb37e14dcb67f` |
| train JSONL | `bb932b756c9bb3f62ca2b3e2b3940b1cac6ae763c08b47168edbe67ed6ea0028` |
| dev JSONL | `85a1b0fe3b69f84b9ec0cfe229f6eca781f31d29cfe0049de1b00b128b4ac7b4` |
| dev predictions | `74585c9b92dd5322568afaaf81e0e2cd802b23d6046b125abf485cc4225ecc89` |
| dev metrics | `7845377b8c9afbb98c277420f245301f13e195a3ab3c51f2697d602fc5c9a081` |

v0.32 的冻结基线清单再次核验以上训练数据、预测、指标、adapter config、adapter 权重和 training report 共 7 项，结果为 7/7 match。

## 8. 报告与复现入口

v0.31 当时已经保存了内容，但没有按 RadarMind 版本文件名归档，因而版本索引曾从 v0.30 直接跳到 v0.32。现将既有资产统一映射如下：

- [完整环境、下载、数据生成、训练和推理指南](reproduction_qwen_vl.md)
- [完整结果与问题族诊断](reproduction_results_v1.md)
- [图文技术报告 PDF](../../reports/drivelm_reproduction_v1/drivelm_reproduction_technical_report.pdf)
- [PDF 构建说明与资产清单](../../reports/drivelm_reproduction_v1/README.md)

本文件是 v0.31 的 RadarMind 规范归档入口；上述原始报告继续作为详细事实来源，不复制或覆盖历史模型。

## 9. 与 v0.32 的关系

v0.31 的单时刻六相机模型是历史 A0。v0.32 在独立目录加入因果 CAM_FRONT 历史帧和问题坐标 ROI，并同时训练新的 uninterrupted A0-control。v0.32 没有覆盖 v0.31 权重；目前历史 A0 仍是总体 EM/F1/ROUGE-L 最好的 checkpoint。
