# v0.41：Qwen3.8-Max API 六路相机零样本评测（764/3,355）

## 状态与边界

本实验将阿里云 OpenAI-compatible 接口实际暴露的 `qwen3.8-max` 作为独立
零样本对照，不修改 B10、MoL-700 或 v0.40 RL 主线。模型对本地
scene-isolated dev 完成 764/3,355 条后，API 返回一周 token-plan 配额耗尽；
因此下列结果是 **22.77% coverage 的部分集结果，不是全量 dev、DriveLM-DS
Final 或官方隐藏榜单分数**。

部分结果按 ID 持续落盘，可在配额恢复或更换 Key 后从第 765 条恢复。API Key
只从环境变量读取，没有写入代码、结果或 Git 历史。

## 推理协议

| 项目 | 配置 |
| --- | --- |
| 模型 | `qwen3.8-max` |
| 输入 | 单帧六路同步相机 + 原始 DriveLM 问题 |
| 每图最大像素 | 100,352（与 B10/MoL 的视觉预算一致） |
| Temperature | 0 |
| 最大可见回答 | 256 tokens |
| 思考模式 | 服务商默认 |
| 已完成 | 764 / 3,355 |
| 部分集任务数 | Perception 199 / Prediction 130 / Planning 329 / Behavior 106 |

请求脚本支持并发、重试、精确 ID coverage 检查和 `.partial.jsonl` 断点恢复。
服务商默认思考模式会产生不可见 reasoning tokens，导致个别请求出现明显长尾；
本实验没有把密钥或原始响应提交到仓库。

## 764 条实测指标

| Task | n | Exact Match | Token-F1 | ROUGE-L |
| --- | ---: | ---: | ---: | ---: |
| Overall | 764 | **15.9686%** | **29.7763%** | **26.0661%** |
| Perception | 199 | 40.7035% | 57.3139% | 51.7293% |
| Prediction | 130 | 0.0000% | 6.2191% | 4.4910% |
| Planning | 329 | 0.0000% | 19.5596% | 15.0047% |
| Behavior | 106 | 38.6792% | 38.6792% | 38.6792% |

同一部分集包含 195 条多选题，准确率为 **62.5641%**。

## 严格同 ID 对比

四个模型只在相同的 764 个 QA ID 上比较：

| Model | EM | Token-F1 | ROUGE-L | MC Accuracy |
| --- | ---: | ---: | ---: | ---: |
| Raw Qwen2.5-VL-7B | 11.2565% | 21.8378% | 19.2737% | 44.1026% |
| **Qwen3.8-Max API zero-shot** | **15.9686%** | **29.7763%** | **26.0661%** | **62.5641%** |
| B10 CE-LoRA | 43.4555% | 73.7279% | 71.6917% | 85.1282% |
| MoL-700 | 43.5864% | 74.9443% | 72.6615% | 84.6154% |

Qwen3.8-Max 相对本地 Raw 7B 的 EM、Token-F1、ROUGE-L 和 MC 分别提高
4.7120、7.9384、6.7924 和 18.4615 个百分点，但仍显著低于 DriveLM 垂域
SFT 与 MoL。主要短板集中在 DriveLM 特有的 Prediction 答案分布、Planning
动作模板以及精确 `<cN,CAMERA,x,y>` grounding，而不是基础图像描述能力。

## 仅供容量规划的全量外推

为修正这 764 条并非随机抽样带来的难度偏差，使用以下同 ID 校准：

```text
Qwen full projection = Raw full observed
                     + (Qwen partial - Raw partial)
```

得到 EM 18.4229%、Token-F1 32.1724%、ROUGE-L 28.5300%、MC 70.1469%。
这些值仅为外推，**不能写入全量 leaderboard，也不能替代实际 3,355 条评测**。
Graph eligibility、语义 judge 和坐标 Match 都是非线性的，因此本阶段不预测
DriveLM-DS Final。

## 复现与继续

推理入口：

```bash
export ALIYUN_TOKEN_PLAN_API_KEY='YOUR_KEY'

python reproduction/qwen_api_v041/infer_api.py \
  --input-jsonl /path/to/qwen_v033_c00_dev.jsonl \
  --output-json /path/to/qwen38max_dev_predictions.json \
  --base-url https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1 \
  --model qwen3.8-max --workers 16 \
  --max-image-pixels 100352 --max-tokens 256 --resume
```

下一次继续必须复用同一 partial 文件和协议。只有得到 3,355/3,355 coverage，
才能运行完整 offline、DriveLM-DS、graph-gating 和与 MoL-700 的正式比较。

逐条预测位于 [`qwen38max_dev_predictions_764.jsonl`](../../results/v041/qwen38max_dev_predictions_764.jsonl)，机器可读摘要位于 [`qwen38max_partial_764.json`](../../results/v041/qwen38max_partial_764.json)。
