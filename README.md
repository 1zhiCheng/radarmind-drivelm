<div align="center">

# RadarMind-DriveLM

**A reproducible six-camera DriveLM pipeline for open-source vision-language driving research**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Qwen2.5-VL](https://img.shields.io/badge/VLM-Qwen2.5--VL-6f42c1)](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
[![Dataset](https://img.shields.io/badge/Dataset-DriveLM--nuScenes-orange)](https://huggingface.co/datasets/OpenDriveLab/DriveLM)
[![CI](https://github.com/1zhiCheng/radarmind-drivelm/actions/workflows/ci.yml/badge.svg)](https://github.com/1zhiCheng/radarmind-drivelm/actions)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

Single-frame six-view perception · Graph-VQA · Qwen2.5-VL LoRA · pure CE SFT · local DriveLM-DS evaluation

</div>

<p align="center">
  <img src="assets/demo/drivelm_preview.gif" width="820" alt="RadarMind DriveLM continuous reasoning demo">
</p>

<p align="center">
  <a href="assets/video/RadarMind_DriveLM_continuous.mp4">32 s demo</a> ·
  <a href="assets/video/RadarMind_DriveLM_scene1094_continuous.mp4">wet-night demo</a> ·
  <a href="reports/drivelm_reproduction_v1/drivelm_reproduction_technical_report.pdf">technical report</a>
</p>

## What this repository adds

The upstream DriveLM release provides data and challenge utilities, but not a complete modern open-weight VLM training path. This repository keeps the official DriveLM-nuScenes v1.1 question IDs, six-camera order and submission schema, and adds a runnable Qwen2.5-VL implementation:

- deterministic scene-isolated train/dev construction with leakage checks;
- six independent camera inputs in the fixed `front / front-left / front-right / back / back-left / back-right` order;
- assistant-token-only autoregressive cross-entropy LoRA SFT;
- single-GPU baseline and multi-GPU `accelerate` training;
- resumable 3,355-QA inference with 100% coverage accounting;
- lexical evaluation plus a cached DeepSeek replacement for the unavailable GPT semantic judge;
- official-val JSON export and continuous DriveLM-style reasoning videos.

This is a modern equivalent reproduction, not a parameter-identical reproduction of the paper's unpublished BLIP-2 training stack.

## Pipeline

<p align="center">
  <img src="reports/drivelm_reproduction_v1/assets/reproduction_pipeline.png" width="900" alt="DriveLM reproduction pipeline">
</p>

```text
DriveLM-nuScenes v1.1 QA + six synchronized images
  -> official extraction and scene-isolated split
  -> Qwen2.5-VL + LoRA, assistant-only CE
  -> 3,355 local-dev predictions
  -> EM / Token-F1 / ROUGE-L / MC accuracy
  -> DriveLM-DS structural proxy
  -> official-val submission JSON
```

The active leaderboard path is camera-only and single-frame. Radar, LiDAR, map, CARLA labels, history frames and dev references are intentionally excluded.

## Results

The reproduced v0.31 Qwen2.5-VL-3B checkpoint covers every item in the fixed local dev split:

| Split / task | QA | Exact Match | Token-F1 | ROUGE-L |
| --- | ---: | ---: | ---: | ---: |
| Overall | 3,355 | **44.26%** | **73.33%** | **71.59%** |
| Perception | 890 | 45.51% | 82.56% | 78.56% |
| Prediction | 599 | 75.13% | 90.41% | 89.05% |
| Planning | 1,399 | 22.16% | 61.76% | 60.72% |
| Behavior | 467 | 68.52% | 68.52% | 68.52% |

Multiple-choice accuracy is **81.46%** over 890 questions. These are local scene-isolated results—not the hidden challenge-server score. The newer B00 controlled run and the v0.35 router are documented in [the pure-CE experiment report](docs/current/VERSION_0_36_DRIVELM_CAMERA_ONLY_PURE_CE.md).

## Quick start

### 1. Environment

PyTorch must be installed separately for the CUDA version on your machine.

```bash
git clone git@github.com:1zhiCheng/radarmind-drivelm.git
cd radarmind-drivelm

conda create -n radarmind-drivelm python=3.11 -y
conda activate radarmind-drivelm
# Install a CUDA-compatible torch build first: https://pytorch.org/get-started/locally/
python -m pip install -r requirements.txt
```

Validated workstation configuration: Python 3.12, PyTorch 2.11, Transformers 5.5, PEFT 0.19. Earlier compatible versions are accepted by `requirements.txt`.

### 2. Download data and model

Never write a Hugging Face token into a script or commit it to Git.

```bash
hf auth login
mkdir -p downloads data models

hf download OpenDriveLab/DriveLM \
  drivelm_nus_imgs_train.zip drivelm_nus_imgs_val.zip \
  v1_1_train_nus.json v1_1_val_nus_q_only.json \
  --repo-type dataset --local-dir downloads

unzip -q downloads/drivelm_nus_imgs_train.zip -d data
unzip -q downloads/drivelm_nus_imgs_val.zip -d data
mkdir -p data/QA_dataset_nus
cp downloads/v1_1_*_nus*.json data/QA_dataset_nus/

hf download Qwen/Qwen2.5-VL-3B-Instruct \
  --local-dir models/Qwen2.5-VL-3B-Instruct
python scripts/check_setup.py
```

The audit must report `six_camera_dirs: 6` and `ready: true` before dataset construction.

### 3. Build the deterministic JSONL

```bash
python reproduction/qwen_vl/build_dataset.py \
  --train-json data/QA_dataset_nus/v1_1_train_nus.json \
  --val-json data/QA_dataset_nus/v1_1_val_nus_q_only.json \
  --output-dir data/reproduction/qwen_vl \
  --seed 42 --dev-ratio 0.1
```

Expected counts are 26,095 train QA and 3,355 scene-isolated dev QA. Image paths are resolved and verified during the build.

### 4. Dry-run before training

```bash
python reproduction/qwen_vl_v036/train_ddp.py \
  --model-path models/Qwen2.5-VL-3B-Instruct \
  --train-jsonl data/reproduction/qwen_vl/qwen_train.jsonl \
  --output-dir /tmp/radarmind-drivelm-dry-run \
  --max-train-samples 64 --max-pixels $((128*28*28)) --dry-run
```

The report must contain exactly six visual inputs per sample and at least one supervised assistant token.

### 5. Train

Single GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python reproduction/qwen_vl/train.py \
  --model-path models/Qwen2.5-VL-3B-Instruct \
  --train-jsonl data/reproduction/qwen_vl/qwen_train.jsonl \
  --output-dir models/radarmind-drivelm-3b \
  --device cuda:0 --batch-size 1 --gradient-accumulation-steps 4 \
  --epochs 1 --learning-rate 2e-4 --save-steps 500
```

Two GPUs with the v0.36 pure-CE trainer:

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --num_processes 2 \
  reproduction/qwen_vl_v036/train_ddp.py \
  --experiment-name radarmind-drivelm-3b \
  --model-path models/Qwen2.5-VL-3B-Instruct \
  --train-jsonl data/reproduction/qwen_vl/qwen_train.jsonl \
  --output-dir models/radarmind-drivelm-3b-ddp \
  --per-device-batch-size 1 --gradient-accumulation-steps 2 \
  --epochs 1 --learning-rate 2e-4 --lora-rank 8 --lora-alpha 16 \
  --max-pixels $((128*28*28)) --max-length 4096 --save-steps 500
```

### 6. Infer and evaluate

```bash
python reproduction/qwen_vl/infer.py \
  --model-path models/Qwen2.5-VL-3B-Instruct \
  --adapter-path models/radarmind-drivelm-3b \
  --input-jsonl data/reproduction/qwen_vl/qwen_dev.jsonl \
  --output-json outputs/dev_predictions.json \
  --device cuda:0 --max-new-tokens 256 --resume

python reproduction/qwen_vl/evaluate_offline.py \
  --references-jsonl data/reproduction/qwen_vl/qwen_dev.jsonl \
  --predictions-json outputs/dev_predictions.json \
  --output-json outputs/dev_metrics.json
```

For DriveLM-DS, store the DeepSeek key outside the repository and restrict its permissions:

```bash
mkdir -p ~/.config/radarmind
printf '%s' "$DEEPSEEK_API_KEY" > ~/.config/radarmind/deepseek_api_key
chmod 600 ~/.config/radarmind/deepseek_api_key

python reproduction/drivelm_ds_eval/evaluate.py \
  --references-jsonl data/reproduction/qwen_vl/qwen_dev.jsonl \
  --predictions-json outputs/dev_predictions.json \
  --output-json outputs/drivelm_ds.json \
  --cache-file outputs/deepseek_judge.sqlite
```

DriveLM-DS preserves the public metric structure but substitutes a frozen DeepSeek semantic judge; report it only as a local proxy.

## Repository layout

```text
challenge/                    official DriveLM extraction and evaluation tools
reproduction/qwen_vl/        dataset, single-GPU SFT, inference and lexical eval
reproduction/qwen_vl_v036/   controlled pure-CE multi-GPU trainer
reproduction/drivelm_ds_eval local structural/semantic proxy evaluator
docs/current/                 current experiment contracts and results
docs/demos/                   continuous-video reproduction notes
reports/                      illustrated Chinese technical report
assets/                       preview images and full MP4 demos
```

## Limitations

- The local split is for reproducible model selection; official val has no public answers.
- The published results use visual key frames rather than raw temporal camera streams.
- Coordinate grounding and planning remain the main error sources.
- Model weights and datasets are intentionally not committed to this repository.

## Acknowledgement and citation

Built on [DriveLM](https://github.com/OpenDriveLab/DriveLM), [nuScenes](https://www.nuscenes.org/) and [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL). Please cite the original projects when using their data, code or models. The upstream DriveLM citation is also available in [CITATION.cff](CITATION.cff).

## License

Code in this repository is released under [Apache-2.0](LICENSE). DriveLM data, nuScenes images and model weights retain their respective licenses and terms.
