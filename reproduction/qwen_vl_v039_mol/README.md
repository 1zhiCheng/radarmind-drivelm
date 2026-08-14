# v0.39 adaptive mixture of LoRA experts

This directory reproduces the v0.39A feasibility control and the v0.39B adaptive checkpoint search. It keeps one LoRA adapter for each official DriveLM task family and routes by the task key already present in the question hierarchy. The router never reads the reference answer.

## Experiment matrix

- `M00`: frozen v0.37B checkpoint-75;
- `M01`: one shared LoRA continued for 100 CE updates;
- `M10` / v0.39A: four task LoRAs continued for 100 CE updates;
- v0.39B: clean four-expert rerun with checkpoint evaluation and automatic early stopping.

M01 and each active M10 expert have the same rank, learning rate, seed, visual budget and update count. M10 uses four adapters in total and therefore has about four times M01's aggregate training compute; only one adapter is active for a question.

## Required inputs

Prepare the camera-only JSONL described in the root README, the v0.37B adapter, and the Qwen2.5-VL-7B base model. The checked-in shell launchers record the exact workstation paths used for the published run. For another machine, change the path block at the top of each launcher; no path is consulted by the Python model or metric implementations themselves.

## Run

Build answer-independent task manifests:

```bash
python reproduction/qwen_vl_v039_mol/build_mol_manifests.py \
  --train-jsonl data/reproduction/qwen_vl/qwen_train.jsonl \
  --dev-jsonl data/reproduction/qwen_vl/qwen_dev.jsonl \
  --output-dir data/reproduction/qwen_vl_v039_mol/manifests
```

The published experiment first ran `run_expert_pilot.sh` plus the shared control, then used the adaptive sequence:

```bash
bash reproduction/qwen_vl_v039_mol/run_expert_sweep.sh
bash reproduction/qwen_vl_v039_mol/run_sweep_evaluation.sh
bash reproduction/qwen_vl_v039_mol/run_adaptive_controller.sh
```

The controller is restart-safe. It treats metric files and prediction IDs as source of truth, resumes partial inference, extends training in 200-step blocks, evaluates every 100 steps, and stops after two checkpoints fail to improve Token-F1 by at least `0.0005`. Coverage, Planning, MC accuracy, graph eligibility and coordinate grounding are safety gates. `1500` is a compute guard, not a target.

Final selection is written to `sweep/best_checkpoint.json`. If the selected longer-run checkpoint fails full DriveLM-DS or the exact same-ID audit, the finalizer points this file back to v0.39A instead of silently promoting it.

## Published result

The controller evaluated steps 100 through 900. Step 700 reached 74.5346% Token-F1; steps 800 and 900 failed the minimum-delta rule, so training stopped and step 700 was restored. Full DriveLM-DS improved from 0.596955 to 0.608245. On 1,812 identical eligible IDs it improved from 0.593502 to 0.607388, and every paired promotion gate passed.

See [`results/v039b/summary.json`](../../results/v039b/summary.json) and the [complete version report](../../docs/current/VERSION_0_39_DRIVELM_ADAPTIVE_MOL.md).
