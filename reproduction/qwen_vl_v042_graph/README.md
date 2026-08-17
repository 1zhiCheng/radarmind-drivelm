# v0.42 frame-level Graph trajectory training

This stage reconnects the official extracted DriveLM nodes at frame level. It
does not invent graph metadata: every one of the 3,605 train and 467 dev frames
already contains Perception, Prediction, Planning and Behavior nodes in a
strict monotonic order. The builder records explicit stage-DAG edges and keeps
the existing scene-isolated split.

The first controlled experiment is multi-turn Graph-SFT. Six synchronized
cameras appear once at the first node; all later questions and answers remain
in one causal conversation. Every assistant span receives ordinary
autoregressive CE, while system, image and user tokens are ignored. Thus later
losses can attend through the representations of earlier answers without
introducing a judge reward or hidden-val leakage.

```bash
PY=/home/zhangzongyuan/anaconda3/envs/radargym-rl/bin/python

$PY reproduction/qwen_vl_v042_graph/build_graph_trajectories.py \
  --train-jsonl /mnt/data/zzy/drivelm/reproduction/qwen_vl_stage2_ablation_v1/qwen_v033_c00_train.jsonl \
  --dev-jsonl /mnt/data/zzy/drivelm/reproduction/qwen_vl_stage2_ablation_v1/qwen_v033_c00_dev.jsonl \
  --output-dir /mnt/data/zzy/drivelm/reproduction/qwen_vl_v042_graph/manifests

bash reproduction/qwen_vl_v042_graph/run_graph_sft.sh
```

Initialization is the promoted shared v0.37B-75 adapter. MoL-700 remains the
frozen full-system baseline because four independently switched adapters do not
permit downstream gradients to update upstream experts through discrete text.
Promotion requires predicted-context graph rollout on all 3,355 dev QA,
complete DriveLM-DS judging and a same-ID comparison against MoL-700.

## Recorded v0.42 results

Two isolated experiments were run from the promoted shared v0.37B-75 adapter.
Both used six cameras once per frame, 3,605 train trajectories, 467 dev
trajectories, three RTX 5090 GPUs and checkpoints every 100 updates.

| Variant | Training records | Selected | Graph dev NLL | EM | Token-F1 | Planning /100 | Coord. F1 | Full Final | Same-ID Final delta | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| v0.42A pure Graph-CE | 3,605 complete graphs | step 600 | **0.36713** | 43.0402% | 73.0446% | 71.2974 | **22.7273%** | **0.609998** | -0.003428 | rejected |
| v0.42B Graph + balanced anchors | 3,605 graphs + 3,604 anchors | step 600 | 0.37045 | 42.6230% | 72.5647% | 70.6977 | 21.4646% | 0.605430 | -0.005678 | rejected |
| frozen MoL-700 | four task adapters | step 700 | n/a | **43.9940%** | **74.5346%** | **72.4769** | 14.6465% | 0.608245 | control | **retained** |

v0.42A raises the candidate-dependent full Final by improving coordinate
matching, but its exact common-eligible audit on 1,848 IDs lowers Final from
0.609563 to 0.606136 and Planning by 2.224 points. v0.42B restores independent
prompt behavior in a fixed 400-QA diagnostic, yet does not restore Planning
when upstream predictions are inserted into the causal context. Its 1,827-ID
paired Final falls from 0.607649 to 0.601971. Neither Graph branch is promoted.

Reproduce v0.42A with `run_graph_sft.sh`; reproduce the deterministic 1:1
Graph/anchor v0.42B control with `run_graph_anchor_sft.sh`. Checkpoint screening
must be followed by 3,355-QA predicted-context rollout and same-ID audit; NLL or
candidate-dependent Final alone is not a promotion criterion.
