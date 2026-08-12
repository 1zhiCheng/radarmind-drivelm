# RadarMind v0.35：DriveLM 单帧六相机刷榜主线

日期：2026-08-11
状态：v0.35 路由基线已完成；v0.36 及以后为重新规划后的执行阶段

## 1. 本版本的决策

当前 DriveLM 刷榜主线只使用：

- DriveLM-nuScenes v1.1 的训练问答；
- 一个当前关键帧；
- 同一时刻的六路相机：`CAM_FRONT`、`CAM_FRONT_LEFT`、
  `CAM_FRONT_RIGHT`、`CAM_BACK`、`CAM_BACK_LEFT`、`CAM_BACK_RIGHT`；
- 问题文本，以及推理过程中由模型自身生成的上游回答。

明确不进入这一主线的内容：历史帧、固定时间偏移帧、毫米波雷达、
激光雷达、HD Map、nuScenes 真值框、未来轨迹标签、CARLA 数据和外部
带答案驾驶数据。nuScenes 原始 camera/radar 下载继续在后台运行，但仅作为
后续多传感器研究储备，不参与当前模型选择与排行榜提交。

这样做的目的不是否定多传感器路线，而是先固定输入域，减少变量，把一个
可复现、可公平比较的 DriveLM camera-only 系统做到尽可能强。

## 2. 数据与评测合同

### 2.1 数据边界

| split | 用途 | 是否包含答案 | 约束 |
| --- | --- | --- | --- |
| DriveLM v1.1 train | SFT、偏好样本构建 | 是 | 只按 scene 切分，不得把 dev scene 混入 |
| scene-isolated dev | 本地模型选择 | 是 | 固定 3,355 QA，不回流训练 |
| official q-only val | 官方提交文件生成 | 否 | 15,480 QA，只推理，不用于调参 |

现有训练集为 26,095 QA，本地 dev 为 3,355 QA。官方 q-only 清单已经核验，
含 15,480 条问题和任务 `tag`：tag 0 为 7,744，tag 1 为 6,599，tag 2 为
799，tag 3 为 338，因此分任务 adapter/router 可以在正式推理时确定性使用，
不需要窥视答案。

### 2.2 官方结构的四项指标

本地使用冻结的 DriveLM-DS 代理评测：保留公开 DriveLM 的计算结构，仅将
原 GPT 语义裁判替换为温度为 0 的 DeepSeek V4 Flash。最终分数为：

```text
Final = 0.4 × Planning + 0.2 × Language + 0.2 × Match + 0.2 × Accuracy
```

- `Accuracy`：tag 0 的精确字符串匹配；
- `Planning`：tag 1 的语义/动作正确性裁判；
- `Language`：tag 2 的 BLEU-1..4、ROUGE-L、CIDEr 组合；
- `Match`：tag 3 坐标 F1 与图语义得分的组合。

DriveLM-DS 只用于本地相对选型，不能写成官方隐藏服务器分数。最终候选必须
生成官方格式结果并由用户确认后提交服务器。

### 2.3 公平性硬约束

每个实验必须同时满足：

1. 3,355/3,355 prediction coverage；
2. 使用同一 reference 文件、同一 evaluator、同一 judge prompt/cache；
3. 报告 eligible、gated-out、每个 tag 的实际样本数；
4. graph eligible 不得低于 C00-CE 的 1,919，避免靠排除难题虚增均分；
5. 同时报告 official-gated、固定共同子集和全量 lexical 指标；
6. 训练失败或模型拒答不得删除样本，必须保留空答案并计为错误；
7. dev 参考答案不得出现在训练 prompt、候选生成或解码规则中。

## 3. v0.35 已完成：无重训任务路由

### 3.1 动机

C00-CE 和 C00-OA 使用相同的 Qwen2.5-VL-3B 基座、相同 26,095 条训练 QA、
相同六路当前帧。两者区别主要是损失权重：OA 提升了封闭式 tag-0 答案，
但牺牲了规划、语言和图语义。继续用一个 adapter 强迫四种任务共享同一损失
会产生梯度冲突，因此先验证分任务路由的上限。

### 3.2 路由规则

```text
tag 0 Accuracy         -> C00-OA
tag 1 Planning         -> C00-CE
tag 2 Language/anchor  -> C00-CE
tag 3 Match/graph      -> C00-CE
```

本地 dev 的 467 个 frame 中，第一条 graph-anchor QA 全部属于 tag 2。路由器
强制这 467 条继续使用 C00-CE，因此 graph gating 与 C00-CE 完全相同。

### 3.3 实测结果

| 模型 | Eligible | Accuracy | Planning | Language | Match | DriveLM-DS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C00-CE | 1,919 | 78.366% | 69.234 | 47.738% | 31.812 | 0.592768 |
| C00-OA | 1,921 | 80.838% | 68.456 | 47.330% | 31.111 | 0.592381 |
| **CE/OA tag router** | **1,919** | **80.787%** | **69.234** | **47.738%** | **31.812** | **0.597609** |

路由版相对 C00-CE 绝对提升 0.004841，约为 0.82% 的相对提升。1,357 条 tag-0
记录切到 OA，其中 109 条答案发生变化；另外 1,998 条保留 CE。所有 791 个
语义裁判结果均命中冻结缓存，没有新增 API 随机性。

产物：

- 路由预测：`$DRIVELM_ROOT/reproduction/qwen_vl_score_optimization_v1/c00_ce_oa_tag_router_predictions.json`
- 构建审计：`$DRIVELM_ROOT/reproduction/qwen_vl_score_optimization_v1/c00_ce_oa_tag_router_build_report.json`
- 完整评测：`$DRIVELM_ROOT/reproduction/qwen_vl_score_optimization_v1/c00_ce_oa_tag_router_drivelm_ds.json`
- 可复现脚本：`DriveLM-main/reproduction/qwen_vl_score_optimization/build_tag_router.py`

## 4. 重新规划后的一个月执行阶段

整体依赖关系如下：

```text
v0.35 冻结路由基线
        |
v0.36 强基座与视觉分辨率
        |
v0.37 分任务 adapter 与结构化损失
        |
v0.38 graph-anchor / 下游图一致性
        |
v0.39 离线偏好优化与可选 GRPO
        |
v0.40 解码集成、官方格式验证与提交
```

### v0.36：强基座与视觉预算（第 1 周）

目标：先判断收益来自模型容量还是图像分辨率，不同时改变其他训练协议。

候选基座按风险排序：

1. Qwen2.5-VL-7B-Instruct：与当前 3B 代码路径兼容，作为主升级；
2. Qwen3-VL-8B-Instruct：空间理解和 grounding 更强，作为第二候选；
3. 当前 Qwen2.5-VL-3B：保留为 launcher/数据的控制组。

视觉预算按每张图的最大视觉 token 做单变量筛选：128（当前）、256、512。
六路视角顺序固定，禁止拼接成一张图后改变空间语义。先用固定训练子集做吞吐、
显存和收敛筛选，再对最好的两个组合使用全部 26,095 QA 训练。

建议第一轮训练配置：

| 参数 | 建议值 |
| --- | --- |
| precision | BF16 |
| optimizer | AdamW |
| epochs | 1.5，保存 0.5/1.0/1.5 epoch checkpoint |
| LoRA | r=32, alpha=64, dropout=0.05 |
| LR | 1e-4 起步，带 3% warmup 与 cosine decay |
| max text length | 4096 |
| effective batch | 固定为 8；所有容量对照保持一致 |
| seed | 42 用于筛选；最终候选补 43、44 |

GPU 分工：三张 RTX 5090 用于训练主实验；A6000 用于 checkpoint 推理与评测。
若混合四卡 DDP 被 A6000 限速，则不强行四卡同步。LoRA 不需要 ZeRO-3；先用
Accelerate DDP。只有全量解冻视觉 merger 或显存不足时才启用 DeepSpeed
ZeRO-2。每次运行必须记录有效 batch、实际更新步数和 tokens/image。

晋级条件：相对 v0.35 路由基线 Final 至少 +0.003，且没有依靠 eligible 数下降；
如果 7B/8B 未超过阈值，就保留 3B 并把算力转给分任务训练。

### v0.37：分任务 adapter 与官方指标对齐损失（第 2 周）

目标：消除 OA 单模型中已经观察到的任务梯度冲突。共享一个冻结基座，但训练
四个轻量 adapter；官方 val 自带 tag，因此推理可以确定性路由。

统一基础损失仍为 assistant token 的自回归交叉熵，但按任务增加不同约束：

```text
L_total = L_token_CE
        + lambda_format * L_format
        + lambda_coord * L_coord
        + lambda_anchor * L_anchor
        + lambda_KL * L_reference_adapter
```

- tag 0：只优化规范化答案 token；使用候选集合约束解码，目标是 exact match；
- tag 1：保留完整语义 CE，增加动作、对象状态关键 token 权重；
- tag 2：作为每帧 important-object anchor，增加对象 ID、相机名和坐标 span 权重；
- tag 3：增加坐标格式损失和坐标 span 的加权 CE；坐标解析失败直接计入格式损失；
- KL/reference 项约束新 adapter 不偏离 v0.36 最优模型，减少语言能力退化。

先做 `shared adapter` 与 `four adapters` 的严格 A/B。禁止把 OA 的加权 CE 与
更大模型、更高分辨率同时改变。最终报告每个 adapter 的数据量、可训练参数量、
训练步数和路由覆盖。

晋级条件：Final 至少 +0.003；Accuracy、Planning、Language、Match 中不得有
任一项绝对下降超过 0.2 个百分点，除非总分提升超过 0.008 且三随机种子一致。

### v0.38：图锚点和下游 QA 一致性（第 3 周前半）

DriveLM 的第一条 important-object 回答决定后续哪些图问题可评测，也是当前
1,436 条 QA 被 gated out 的来源。这里仍只看同一组六路图像，不加入时间帧。

训练流程：

1. 按 scene 做 K-fold，在 train 上产生 out-of-fold 第一问预测；
2. 解析模型预测的对象 ID、相机与坐标，形成文本 graph memory；
3. 下游 QA 的输入增加该模型生成 memory，而不是参考答案；
4. 训练早期混合 gold anchor 与 OOF predicted anchor，随后逐步提高 predicted
   比例，避免训练/推理暴露偏差；
5. 推理时先回答第一问，再把自身结果传给同帧后续 QA。

不能直接把 dev 或 official val 的参考第一问注入 prompt。模型生成的文本记忆
属于 agent 内部状态，不改变“单帧六相机”传感器约束。

晋级除了 Final 外还要求：eligible 不少于 1,919；第一问对象坐标 precision、
recall、F1 均报告；在 C00-CE 固定共同 eligible 子集上分数不能退化。

### v0.39：离线偏好优化，达标后再做 GRPO（第 3 周后半）

不直接从 PPO/GRPO 开始。原因是四项官方指标包含不可微字符串、语言指标、
坐标匹配和外部语义裁判，未经校准的在线 reward 很容易被格式或 gating 投机。

标准顺序：

1. 从 v0.38 最优模型对 train 问题生成 2--4 个候选；
2. 用确定性规则计算 Accuracy、格式、坐标和语言分；
3. 对 planning/graph 候选使用冻结 DeepSeek rubric 排序；
4. 构建 chosen/rejected 对，先训练 DPO 或 SimPO；
5. 保留 SFT anchor/KL，防止 reward optimization 破坏六相机 grounding；
6. 只有当本地 reward 与冻结 DriveLM-DS 在保留集上的 Spearman 相关性 >= 0.7，
   才允许进入短程 GRPO。

GRPO 的 reward 必须按 tag 分开归一化，不能让样本数更多的 tag 0/1 淹没 tag
2/3。PPO、GSPO、SAPO 等只作为 GRPO 不稳定时的后续消融，不在一个月主线
中并行铺开。先获得可靠偏好数据，通常比更换 RL 算法名称更有价值。

### v0.40：推理时优化与官方提交（第 4 周）

推理阶段按 tag 使用独立生成策略：

- tag 0：低温/贪心、候选答案约束、去除解释性尾巴；
- tag 1：保留完整语义回答，禁止简单截断；
- tag 2：对象列表 schema 校验，坐标与相机名强约束；
- tag 3：图答案 schema 与坐标 parser 校验；
- 解析失败时只允许基于模型 logits 的合法格式回退，不可访问参考答案。

候选集成顺序为：单 checkpoint、同模型多 checkpoint、分 tag adapter/router。
只有本地 3,355 QA 全量评测和三 seed 复验通过的候选才生成 15,480 条官方
q-only 预测。提交前执行 ID 集合、顺序、空答案、JSON schema 和 SHA-256 审计；
向官方服务器提交属于外部状态变更，需要用户最后确认。

## 5. 一个月里程碑与资源预算

| 时间 | 交付物 | 退出条件 |
| --- | --- | --- |
| Day 1--2 | v0.35 router、评测审计、固定基线 | 已完成，0.597609 |
| Day 3--8 | v0.36 7B/8B 与 128/256/512 视觉预算 | 选出不靠 gating 的最优基座 |
| Day 9--15 | v0.37 四任务 adapter 与损失消融 | 至少一个候选稳定超过 v0.36 |
| Day 16--20 | v0.38 OOF graph memory | coverage 与共同子集均不退化 |
| Day 21--25 | v0.39 DPO/SimPO；达标才短程 GRPO | reward 相关性和三 seed 通过 |
| Day 26--28 | v0.40 解码、路由和 ensemble | 选出 1--2 个正式候选 |
| Day 29--30 | 官方格式审计、提交、完整报告 | 用户确认后提交，冻结版本 |

模型权重、训练日志和大体积预测统一放在 `$DRIVELM_ROOT`；代码与小型
配置放在 `DriveLM-main/reproduction`；每个版本在 `drivelm/DriveLM-main/docs/current` 保留
中文详尽说明。任何实验都不得覆盖 C00-CE、C00-OA 或 v0.35 的现有产物。

## 6. 下一步实际执行顺序

1. 冻结 v0.35 结果和哈希，补自动对比表生成器；
2. 在当前数据上重跑一个 3B/统一 launcher 控制实验，验证新 DDP pipeline；
3. nuScenes 原始下载继续，不加入训练；
4. 下载 Qwen2.5-VL-7B 到 `$DRIVELM_ROOT/models`；
5. 做 7B 的 128-token/image smoke test，再做 256-token/image 对照；
6. 只有 smoke 的输入形状、loss、显存和 64 条推理均正常，才启动全量训练。

这条顺序优先降低工程风险：先保证评测和数据不变，再扩大模型与图像预算；
随后才改变损失和图结构，最后进行偏好/RL。每阶段都能单独回滚到前一最优版本。
