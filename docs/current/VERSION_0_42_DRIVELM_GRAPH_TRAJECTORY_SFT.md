# v0.42 DriveLM 四阶段 Graph trajectory SFT

## 1. 目标与边界

本版本把同一帧内的 DriveLM QA 按因果顺序组织为：

```text
六路相机 -> Perception -> Prediction -> Planning -> Behavior
```

每个后续节点可看到前面节点的回答。训练时使用 teacher forcing；最终 dev
rollout 强制使用模型自己的上游预测，不允许读取 reference。实验继续保持单帧、
六路相机、camera-only 和 scene-isolated dev，不引入雷达、LiDAR、历史帧或 dev
答案训练。

## 2. 数据审计

| split | frame trajectories | QA nodes | Perception | Prediction | Planning | Behavior |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 3,605 | 26,095 | 6,866 | 4,811 | 10,813 | 3,605 |
| dev | 467 | 3,355 | 890 | 599 | 1,399 | 467 |

每条轨迹均含四种任务且任务顺序单调，节点数为 5–8；train/dev scene
交集为 0。原始数据的 `con_up/con_down` 为空，因此这里只记录相邻阶段 DAG，
不伪造官方未提供的细粒度边。

## 3. 训练方案

### v0.42A：纯 Graph-CE

- 初始化：v0.37B conservative DPO step 75 shared adapter；
- 六路图像只在第一个 user turn 编码一次；
- 所有 assistant spans 计算普通 autoregressive token CE；
- system、image、user 和 padding token 的 label 为 `-100`；
- 三张 RTX 5090，effective trajectory batch 6，BF16；
- LR `1e-6`，600 updates，100-step checkpoint；
- 训练时长 1,080.74 秒（恢复段），约 2.16 秒/update。

损失为：

```text
L_graph = - sum(valid assistant token log-prob) / valid token count
```

它不是离散输出之间的可微反向传播，而是共享 adapter 在一条 causal
conversation 中同时接收四阶段 token loss；下游 loss 可以通过 attention
更新共享参数和上游 token representation。

### v0.42B：Graph + task-balanced CE anchor

v0.42A 的 grounding 提升但 Planning 退化，因此从同一个 v0.37B-75 重新初始化，
不在失败模型上续训。训练 manifest 由以下记录 1:1 混合：

- 3,605 条完整四阶段 Graph trajectory；
- 3,604 条单节点 anchors，四类任务各 901 条。

仍只使用 CE，不使用 judge、reward、dev score 或 DPO。共训练 600 updates，
耗时 1,125.10 秒，平均 1.875 秒/update。

## 4. checkpoint 选择

所有 checkpoint 都在 467 条完整 dev trajectory、3,355 个 QA、116,445 个
监督 token 上计算 teacher-forced NLL。NLL 只用于筛点，不用于晋级。

| step | v0.42A NLL | v0.42B NLL |
| ---: | ---: | ---: |
| 100 | 0.37733 | 0.37986 |
| 200 | 0.37330 | 0.37697 |
| 300 | 0.37088 | 0.37454 |
| 400 | 0.36931 | 0.37278 |
| 500 | 0.36806 | 0.37173 |
| 600 | **0.36713** | **0.37045** |

两条路线都选择 step 600。v0.42A 相对初始化 NLL `0.38799 -> 0.36713`
（-5.38%）；v0.42B 为 `0.38799 -> 0.37045`（-4.52%）。

## 5. 严格 generated-context 全量结果

三个 GPU shard 合并后均为 3,355/3,355、missing 0、extra 0。下游上下文只含
模型预测。DeepSeek-V4-Flash judge 均完整结束且 failures 为 0。

| Metric | MoL-700 | v0.42A | v0.42B |
| --- | ---: | ---: | ---: |
| EM | **43.9940%** | 43.0402% | 42.6230% |
| Token-F1 | **74.5346%** | 73.0446% | 72.5647% |
| ROUGE-L | **72.4050%** | 71.0026% | 70.5456% |
| MC accuracy | **84.4944%** | 83.7079% | 83.4831% |
| Planning judge /100 | **72.4769** | 71.2974 | 70.6977 |
| Coordinate F1 | 14.6465% | **22.7273%** | 21.4646% |
| Match /100 | 30.7513 | 34.0947 | **34.1263** |
| DriveLM-DS Final | 0.608245 | **0.609998** | 0.605430 |

v0.42A 的 candidate-dependent Final 略高，但不同模型的 graph gating 会产生
不同 eligible IDs，不能据此直接晋级。

## 6. same-ID 公平审计

| Audit | MoL-700 Final | Graph Final | Delta | Planning delta | MC delta | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| v0.42A，1,848 common IDs | 0.609563 | 0.606136 | -0.003428 | -2.2240 | -1.2478pp | no |
| v0.42B，1,827 common IDs | 0.607649 | 0.601971 | -0.005678 | -2.4507 | -1.6304pp | no |

两者 judge 都 100% 完成。v0.42A/B 只通过 coordinate regression gate，未通过
Final、Planning 和 MC gate，因此 **MoL-700 继续作为完整系统晋级基线**。

## 7. 结论与下一步

本版本真正跑通了完整四阶段 Graph 训练和无泄漏 rollout，而不是只在 prompt 中
拼接 reference。实验说明：

1. joint Graph-CE 明显改善对象坐标 grounding 和 Match；
2. teacher-forced 上游答案与推理时预测答案之间存在 exposure mismatch；
3. 单节点 CE anchor 能改善独立 prompt 的 Behavior/MC，却不能修复带噪上游上下文
   下的 Planning；
4. 下一轮应做 train-only upstream prediction cache、scheduled sampling/DAgger，
   并对 Planning 增加“正确上游/预测上游”双上下文一致性 CE，而不是继续增加
   teacher-forced steps。

## 8. 复现入口与产物

```bash
bash reproduction/qwen_vl_v042_graph/run_graph_sft.sh
bash reproduction/qwen_vl_v042_graph/run_checkpoint_nll.sh
bash reproduction/qwen_vl_v042_graph/run_graph_anchor_sft.sh
pytest -q reproduction/qwen_vl_v042_graph/test_graph_pipeline.py
```

主要本地结果：

```text
/mnt/data/zzy/drivelm/models/qwen2.5-vl-7b-drivelm-v042-graph-sft/checkpoint-600
/mnt/data/zzy/drivelm/models/qwen2.5-vl-7b-drivelm-v042b-graph-anchor/checkpoint-600
/mnt/data/zzy/drivelm/reproduction/qwen_vl_v042_graph/
/mnt/data/zzy/drivelm/reproduction/qwen_vl_v042_graph_anchor/
```
