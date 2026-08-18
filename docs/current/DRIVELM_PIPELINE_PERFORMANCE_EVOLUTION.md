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
gating、坐标 grounding 和语义 judge。v0.40 最初只在 1,399 条 Planning
trajectory 上选点，随后已将两种 RL policy 接回四专家 router，补齐完整
3,355-QA DriveLM-DS 和 same-ID 审计。

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
      <td>3,355 全任务</td>
      <td>13.7109%</td><td>24.2340%</td><td>21.7376%</td><td>51.6854%</td><td>36.8308</td><td>0.257961</td><td>原始起点</td>
      <td>zero-shot control</td>
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
      <td>3,355 全任务</td>
      <td>43.5469%</td><td>74.0030%</td><td>71.9047%</td><td>84.4944%</td>
      <td>72.0910</td><td>0.606702</td><td>-0.001543</td>
      <td>✗ 全系统未晋级</td>
    </tr>
    <tr>
      <td>GSPO-90</td>
      <td>3,355 全任务</td>
      <td>43.7854%</td><td>74.0246%</td><td>71.9357%</td><td>84.4944%</td>
      <td>72.0756</td><td>0.606640</td><td>-0.001605</td>
      <td>✗ 全系统未晋级</td>
    </tr>
    <tr>
      <td rowspan="2">Graph trajectory SFT</td>
      <td>v0.42A pure Graph-CE</td><td>3,355 predicted-context</td>
      <td>43.0402%</td><td>73.0446%</td><td>71.0026%</td><td>83.7079%</td>
      <td>71.2974</td><td>0.609998</td><td>same-ID -0.003428</td><td>✗ 未晋级</td>
    </tr>
    <tr>
      <td>v0.42B Graph + CE anchors</td><td>3,355 predicted-context</td>
      <td>42.6230%</td><td>72.5647%</td><td>70.5456%</td><td>83.4831%</td>
      <td>70.6977</td><td>0.605430</td><td>same-ID -0.005678</td><td>✗ 未晋级</td>
    </tr>
    <tr>
      <td>InternVL-inspired ensemble</td><td><strong>v0.43 Graph anchor + MoL downstream</strong></td>
      <td>3,355 fixed routing</td><td>43.9940%</td><td>74.4813%</td><td>72.3789%</td><td>84.4944%</td>
      <td><strong>73.0197</strong></td><td><strong>0.612293</strong></td><td><strong>+0.004048</strong></td><td><strong>🏆 晋级</strong></td>
    </tr>
  </tbody>
</table>

完整系统中 MoL/GRPO/GSPO 的 perception、prediction、behavior 与 graph
eligibility 完全相同，均为 1,911；只替换 Planning 答案。因此 Final 下降不是
gating 子集变化造成的，而是 trajectory policy 在 graph-eligible Planning
子集上的泛化没有超过原 MoL Planning expert。

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

- 原始 7B zero-shot 到垂域 SFT，Final：`0.257961 -> 0.594636`，
  绝对提升 `+0.336675`，相对提升约 `+130.52%`。
- 从 7B SFT 到 conservative DPO，Final：`0.594636 -> 0.596356`。
- 从 DPO 到 adaptive MoL，Final：`0.596356 -> 0.608245`。
- 从 7B SFT 到 MoL，Final 累计提高 `+0.013609`，相对提高约 `+2.29%`。
- 在 Planning trajectory 子协议中，GSPO-90 优于 MoL control 与 GRPO-70。
- 完整协议中 GRPO/GSPO Final 为 0.606702/0.606640，均低于 MoL 0.608245，
  所以 MoL-700 在 v0.40 阶段继续作为全系统主线。
- v0.42A/B 完成真正的 P→Prediction→Planning→Behavior 串联训练和无 gold
  predicted-context rollout；A 的全量 Final 为 0.609998，但在 1,848 个 same-ID
  上下降 0.003428，B 在 1,827 个 same-ID 上下降 0.005678，均不晋级。
- Graph-CE 将 coordinate F1 从 14.65% 提到 22.73%，但 Planning 下降；下一步
  应解决 teacher forcing 与预测上游上下文之间的 exposure mismatch。
- v0.43 固定使用 Graph-A 生成每帧 anchor、MoL-700 生成全部 downstream QA；全量 Final 提升到 `0.612293`，1,848-ID paired Final 同样提升 `+0.000993`，因此成为新的全系统主线。

## 5. 补齐实验验收

- [x] Raw 7B：3,355/3,355 coverage，599/599 judge，0 failures。
- [x] GRPO-70：3,355/3,355 coverage，780/780 judge，0 failures。
- [x] GSPO-90：3,355/3,355 coverage，780/780 judge，0 failures。
- [x] 两个 RL 候选与 MoL 的 graph-eligible ID 完全相同：1,911/1,911。
- [x] same-ID 审计确认两者 Final 都严格下降，均不通过全系统晋级门槛。
- [x] v0.42A/B：3,355/3,355 predicted-context coverage，judge 0 failures，same-ID 均未通过。
- [x] v0.43：3,355/3,355 coverage，791/791 cache-only judge；全量与 same-ID Final 均严格提高，全部 paired gates 通过。
