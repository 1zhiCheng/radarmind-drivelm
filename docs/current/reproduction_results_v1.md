# DriveLM 六视角 Qwen-VL v1 结果报告

## 版本结论

本版本完成了 DriveLM-nuScenes v1.1 官方基础 QA 协议下的六相机训练、场景隔离开发集推理和离线评估。它是可运行的 Qwen2.5-VL-3B LoRA 复现基线，不等同于未完整公开的论文 BLIP-2 模型，也不使用官方隐藏 val 答案。

## 训练配置

| 项目 | 数值 |
|---|---|
| 基础模型 | Qwen2.5-VL-3B-Instruct |
| 训练场景 / QA | 619 / 26,095 |
| 开发场景 / QA | 77 / 3,355 |
| 输入 | 六个同步环视相机 |
| 每图最大视觉预算 | 75,264 pixels，约 96 个视觉单元 |
| LoRA 参数 | 18,576,384（0.4923%） |
| batch size | 4 |
| 总更新步 | 6,524 |
| 最终续训段平均 loss | 0.341855 |

训练曾在第 3,000 步随终端会话退出，之后从该 LoRA checkpoint 恢复至 6,524 步。模型参数连续，但旧 checkpoint 未保存 Adam 动量，恢复段优化器状态重新初始化。最终训练报告位于：

```text
$DRIVELM_ROOT/models/qwen2.5-vl-3b-drivelm-sixcam/training_report.json
```

## 完整开发集指标

3,355 条预测全部匹配，coverage 为 100%。以下是本项目的无外部 API 离线指标，不是挑战服务器分数。

| 任务 | 数量 | Exact Match | Token-F1 | ROUGE-L |
|---|---:|---:|---:|---:|
| Overall | 3,355 | 44.26% | 73.33% | 71.59% |
| Perception | 890 | 45.51% | 82.56% | 78.56% |
| Prediction | 599 | 75.13% | 90.41% | 89.05% |
| Planning | 1,399 | 22.16% | 61.76% | 60.72% |
| Behavior | 467 | 68.52% | 68.52% | 68.52% |

多选题共 890 条，准确率为 81.46%。

## 问题族诊断

| 问题族 | 数量 | Exact Match | Token-F1 |
|---|---:|---:|---:|
| Binary reasoning | 467 | 96.36% | 96.36% |
| Moving-status MC | 423 | 95.74% | 95.74% |
| Candidate actions | 466 | 11.59% | 76.69% |
| Important objects | 467 | 0.00% | 70.63% |
| Notice graph | 132 | 0.00% | 69.34% |
| Safe actions | 467 | 19.06% | 59.60% |
| Collision reasoning | 466 | 35.84% | 48.98% |
| Behavior MC | 467 | 68.52% | 68.52% |

对象标识 `<c*,CAM_*,x,y>` 的简化结构匹配 precision 为 30.98%、recall 为 32.82%、F1 为 31.87%。这说明模型能生成 DriveLM 风格答案，但在未见场景中仍难以把任意对象编号、相机方向和精确位置绑定正确。

## 主要问题

1. Planning 明显弱于 perception/prediction，尤其容易把“应停车”错误判断为“保持速度”。
2. 每张图约 96 个视觉单元对车辆类型、行人属性和像素坐标接地过于粗糙。
3. Important-object 和 notice-graph 答案较长，即使语义接近，任意一个对象类别、顺序或 ID 错误都会降低结构可靠性。
4. Behavior 多选仍有约 31.5% 错误，说明仅凭单帧六视图预测自车速度/转向存在观测歧义，后续应引入时序帧或 CAN/ego state。

## 可复核产物

| 产物 | SHA-256 |
|---|---|
| 最终 LoRA adapter | `1473386e2f1d45b1d269c60216d064ebfff42a62f7d3418bd40cb37e14dcb67f` |
| train JSONL | `bb932b756c9bb3f62ca2b3e2b3940b1cac6ae763c08b47168edbe67ed6ea0028` |
| dev JSONL | `85a1b0fe3b69f84b9ec0cfe229f6eca781f31d29cfe0049de1b00b128b4ac7b4` |
| dev predictions | `74585c9b92dd5322568afaaf81e0e2cd802b23d6046b125abf485cc4225ecc89` |

指标和错误分析位于：

```text
$DRIVELM_ROOT/reproduction/qwen_vl/dev_metrics.json
$DRIVELM_ROOT/reproduction/qwen_vl/dev_error_analysis.json
```

## 官方 val 生成结果

无答案的官方 val 已完成全部 15,480 条生成。输出 ID 与 `qwen_val_questions.jsonl` 逐项同序一致，唯一 ID 为 15,480，空答案为 0，schema 全部严格为 `id + answer`。

```text
$DRIVELM_ROOT/reproduction/qwen_vl/official_val_output.json
SHA-256: 77bdaca3d8c777020b9f043ce607557e5275aef7929048f49ee15a8f17cb2185
size: 2,747,936 bytes
```

其中 1,543 条回答被规范化为单个 A/B/C/D 字母。官方 val 不公开真实答案，因此该文件只能通过 DriveLM 挑战服务器获得官方指标。

## v2 方向

已从 train 构建 20,437 条困难任务课程，dev 泄漏为 0。collision 和 notice-graph 各重复 2×，同时保留 safe-actions、behavior 和 important-objects。v2 将从 v1 adapter 继续训练，并测试更高视觉分辨率与目标区域裁剪；评估必须继续使用相同的 77 个隔离场景，比较 planning F1、对象 ID F1 和多选准确率，不能只比较训练 loss。
