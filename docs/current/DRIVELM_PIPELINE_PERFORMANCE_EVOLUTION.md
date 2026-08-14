# RadarMind-DriveLM 全流程性能演进

## 1. 先解释 Trajectory reward

`Trajectory reward` 是 v0.40 为 Planning policy 训练设计的本地、确定性奖励：

```text
R = 0.40 × Token-F1
  + 0.20 × ROUGE-L
  + 0.25 × Action-F1
  + 0.05 × Exact Match
  + 0.05 × Grounding Validity
  + 0.05 × Format Validity
```

它回答的是：“模型生成的规划答案在内容、动作、对象引用和格式上有多好？”它可在
每个 rollout 后立即计算，不调用 API，适合 GRPO/GSPO 的在线优化和 checkpoint
选择。

`DriveLM-DS Final` 回答的是另一个问题：“完整 DriveLM Graph-VQA Agent 在所有
任务上的综合能力有多好？”本项目的本地代理沿用公开结构：

```text
Final = 0.4 × Planning + 0.2 × Language
      + 0.2 × Match + 0.2 × Accuracy
```

它依赖完整 3,355 条 Perception、Prediction、Planning、Behavior 输出、graph
gating、坐标 grounding 和语义 judge。v0.40 只替换 Planning policy，并仅在
1,399 条 Planning trajectory 上评测，所以当时只能报告 Trajectory reward，
不能诚实地填写新的 DriveLM-DS Final。

## 2. 全任务累计晋级链

下表中只有相同的 3,355-QA scene-isolated dev 才比较 DriveLM-DS Final。
`🏆` 表示该阶段正式晋级并成为下一阶段初始化；`✗` 表示完成实验但未晋级。

<table>
  <thead>
    <tr>
      <th>阶段</th>
      <th>候选</th>
      <th>评测范围</th>
      <th>EM</th>
      <th>Token-F1</th>
      <th>ROUGE-L</th>
      <th>MC Acc.</th>
      <th>Planning /100</th>
      <th>DriveLM-DS Final</th>
      <th>相对上一晋级者</th>
      <th>结论</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>原始基模</td>
      <td>Qwen2.5-VL-7B-Instruct zero-shot</td>
      <td>尚未正式评测</td>
      <td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td>
      <td>待补基模对照</td>
    </tr>
    <tr>
      <td>SFT</td>
      <td><strong>B10：7B CE-LoRA</strong></td>
      <td>3,355 全任务</td>
      <td>43.4575%</td><td>73.0000%</td><td>71.0756%</td><td>83.8202%</td>
      <td>70.6348</td><td><strong>0.594636</strong></td><td>首个 7B 垂域基线</td>
      <td><strong>🏆 晋级</strong></td>
    </tr>
    <tr>
      <td rowspan="2">DPO</td>
      <td>v0.37A standard DPO-100</td>
      <td>3,355 全任务</td>
      <td>43.55%</td><td>73.17%</td><td>71.22%</td><td>84.27%</td>
      <td>69.75</td><td>0.59430</td><td>-0.00034</td><td>✗ 未晋级</td>
    </tr>
    <tr>
      <td><strong>v0.37B conservative DPO-75</strong></td>
      <td>3,355 全任务</td>
      <td>43.5768%</td><td>73.1231%</td><td>71.2034%</td><td>84.1573%</td>
      <td>70.8571</td><td><strong>0.596356</strong></td>
      <td><strong>+0.001721（+0.29%）</strong></td><td><strong>🏆 晋级</strong></td>
    </tr>
    <tr>
      <td>Mixture of LoRA</td>
      <td><strong>v0.39B MoL step 700</strong></td>
      <td>3,355 全任务</td>
      <td>43.9940%</td><td>74.5346%</td><td>72.4050%</td><td>84.4944%</td>
      <td>72.4769</td><td><strong>0.608245</strong></td>
      <td><strong>+0.011889（+1.99%）</strong></td><td><strong>🏆 晋级</strong></td>
    </tr>
    <tr>
      <td rowspan="2">Trajectory RL</td>
      <td>GRPO-70</td>
      <td>1,399 Planning</td>
      <td>17.5840%</td><td>56.4231%*</td><td>55.5712%*</td><td>—</td>
      <td>71.4582</td><td>待全协议评测</td><td>Reward +0.003556</td>
      <td>✓ 通过门槛</td>
    </tr>
    <tr>
      <td><strong>GSPO-90</strong></td>
      <td>1,399 Planning</td>
      <td>18.1558%</td><td>56.5607%*</td><td>55.7057%*</td><td>—</td>
      <td><strong>71.5904</strong></td><td>待全协议评测</td>
      <td><strong>Reward +0.004482</strong></td><td><strong>🏆 Planning 晋级</strong></td>
    </tr>
  </tbody>
</table>

\* RL 表中的 Token-F1/ROUGE-L 来自 trajectory reward 的 normalization；与旧版
全任务离线 evaluator 数值不可直接相减。RL 的直接 control 是同一 trajectory
prompt 下的 MoL Planning adapter，而不是上表 MoL 全协议的 graph-gated Planning。

## 3. RL 内部严格同口径比较

<table>
  <thead>
    <tr>
      <th>阶段</th><th>模型</th><th>Trajectory reward</th><th>Token-F1 reward</th>
      <th>ROUGE-L reward</th><th>Action-F1</th><th>Exact</th>
      <th>DeepSeek Planning</th><th>结论</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3">Planning trajectory</td>
      <td>MoL control</td><td>60.8118</td><td>55.8800</td><td>55.0403</td>
      <td>66.3474</td><td>17.2981</td><td>71.2795</td><td>冻结基线</td>
    </tr>
    <tr>
      <td>GRPO-70</td><td>61.1674</td><td>56.4231</td><td>55.5712</td>
      <td><strong>66.4189</strong></td><td>17.5840</td><td>71.4582</td>
      <td>✓ 全门槛通过</td>
    </tr>
    <tr>
      <td><strong>GSPO-90</strong></td><td><strong>61.2600</strong></td>
      <td><strong>56.5607</strong></td><td><strong>55.7057</strong></td>
      <td>66.3474</td><td><strong>18.1558</strong></td>
      <td><strong>71.5904</strong></td><td><strong>🏆 晋级</strong></td>
    </tr>
  </tbody>
</table>

GSPO-90 相对 MoL trajectory control 的 reward、Exact 和 Planning judge 分别提高
0.4482、0.8578、0.3109 个百分点。GRPO 也全面通过冻结门槛，但 GSPO 在主排序
Planning judge 和次排序 reward 上均更高。

## 4. 可以得出的累计结论

- 从 7B SFT 到 conservative DPO，Final：`0.594636 -> 0.596356`。
- 从 DPO 到 adaptive MoL，Final：`0.596356 -> 0.608245`。
- 从 7B SFT 到 MoL，Final 累计提高 `+0.013609`，相对提高约 `+2.29%`。
- 在 Planning trajectory 子协议中，GSPO-90 优于 MoL control 与 GRPO-70。
- 现在不能宣称“RL 将 Final 提升到某个数”，因为该全协议实验尚未完成。

## 5. 补齐最终单表所需实验

1. 对未微调 Qwen2.5-VL-7B 运行 3,355-QA zero-shot inference 与 DriveLM-DS，
   填入原始基模行。
2. 将 GRPO-70、GSPO-90 分别接回四专家 router，只替换 Planning adapter。
3. 对两者运行完整 3,355-QA inference、100% judge-complete DriveLM-DS。
4. 对 MoL/GRPO/GSPO 计算共同 graph-eligible ID 的 same-ID 公平性审计。
5. 只有通过 Final、Planning、coverage、judge completeness 和 same-ID gates，
   才把 GSPO 从“Planning 晋级”升级为“全系统正式晋级”。

完成这组测试后，表内所有主线阶段都能用同一个 DriveLM-DS Final 纵向比较。
