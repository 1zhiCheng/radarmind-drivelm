# RadarMind-DriveLM v0.37B: 任务均衡与 CE 锚定的保守 DPO

状态：实验契约已确定，尚未训练。

## 目标

在不牺牲 B10 planning、开放文本和坐标 grounding 的前提下，验证偏好训练能否保留 v0.37A 带来的多选与 behavior 增益。

## 单变量路线

v0.37B 仍使用单 LoRA，从 B10 重新初始化。数据、模型、视觉预算、dev 和评测协议不变，仅修改后训练采样与目标：

1. 四任务均衡采样，每个 epoch 中 perception、prediction、planning、behavior 各占 25%；
2. 总损失为 DPO loss 加 lambda 乘 chosen-answer CE；
3. 初始 beta=0.05、learning rate=1e-6、lambda=0.1；
4. 保存 25/50/75/100 step，最多不超过 100 step；
5. 以离线 EM、Token-F1、ROUGE-L、MC 和 planning slice 选一个候选；
6. 只有候选同时不低于 B10 的 Token-F1、planning Token-F1，并提高 MC，才运行完整 DeepSeek judge。

## 晋级门槛

- prediction coverage 为 100%；
- DriveLM-DS judge 完成率为 100%；
- Final 必须严格高于 0.59464；
- planning /100 回退不超过 0.5；
- coordinate F1 回退不超过 0.5 percentage point；
- MC accuracy 不低于 B10。

任何门槛失败都保留 B10，不继续 GRPO。

## MoL 条件

本阶段不启用 Mix of LoRA。只有保守单 LoRA 仍呈现明确任务冲突，并且至少两个任务在不同 checkpoint 上获得可重复的最佳点，才建立 MoL 对照。届时需要固定相同偏好数据与损失，仅比较：

- shared single LoRA；
- task-specific LoRA experts with oracle task routing；
- learned router。

先使用 oracle task routing 建立上界，再决定是否值得训练路由器。
