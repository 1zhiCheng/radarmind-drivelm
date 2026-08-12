# DriveLM-nuScenes 六视角复现（Qwen2.5-VL）

## 1. 目标与边界

本复现已经打通：DriveLM v1.1 标注 → 官方基础 QA 抽取 → 六相机 SFT JSONL → Qwen2.5-VL LoRA → 开发集生成 → 离线评估 → 官方 val 输出。

论文 DriveLM-Agent 使用 BLIP-2，但仓库没有完整训练实现。`challenge/` 发布的是 LLaMA-Adapter v2 基线，它需要申请制 LLaMA-1 7B 权重和旧版 adapter checkpoint，本机没有这些受限权重。因此这里忠实保留官方数据规则、六相机输入、问题 ID 与提交格式，以本地 Qwen2.5-VL-3B-Instruct 实现可真正运行的现代等价基线，不把它误称为逐参数论文复现。

## 2. 数据目录与实测统计

```text
$DRIVELM_ROOT/
├── data/QA_dataset_nus/{v1_1_train_nus.json,v1_1_val_nus_q_only.json}
├── data/nuscenes/samples/CAM_*/...
├── reproduction/qwen_vl/
│   ├── official_extracted.json
│   ├── official_train_eval.json
│   ├── official_train_llama.json
│   ├── qwen_train.jsonl
│   ├── qwen_dev.jsonl
│   ├── qwen_val_questions.jsonl
│   └── dataset_report.json
└── models/qwen2.5-vl-3b-drivelm-sixcam-smoke/
```

| 划分 | 场景数 | QA 数 | 答案 | 用途 |
|---|---:|---:|---|---|
| train | 619 | 26,095 | 有 | LoRA SFT |
| dev | 77 | 3,355 | 有 | 按场景隔离的本地评估 |
| official val | 149 | 15,480 | 无 | 挑战提交 |

当前 v1.1 经官方脚本抽取后是 29,450 QA，比 README 所写 29,448 多 2 条。构建器固定 `seed=42`，避免官方多选项随机打乱导致每次答案字母不同。共检查 29,226 张唯一图片，缺失 0 张。

## 3. 环境安装

当前验证环境为 Python 3.12、PyTorch 2.11、Transformers 5.5.4、PEFT 0.19.1：

```bash
conda activate radargym-rl
cd /path/to/radarmind-drivelm
python -m pip install -r reproduction/qwen_vl/requirements.txt
```

基础模型位于 `$DRIVELM_ROOT/models/Qwen2.5-VL-3B-Instruct`。

### 从 Hugging Face 重新下载

不要把 token 写进脚本或 Git。首次机器登录使用 `hf auth login`，然后执行：

```bash
mkdir -p $DRIVELM_ROOT/downloads
hf download OpenDriveLab/DriveLM \
  drivelm_nus_imgs_train.zip drivelm_nus_imgs_val.zip \
  v1_1_train_nus.json v1_1_val_nus_q_only.json \
  --repo-type dataset --local-dir $DRIVELM_ROOT/downloads

mkdir -p $DRIVELM_ROOT/data
unzip -q $DRIVELM_ROOT/downloads/drivelm_nus_imgs_train.zip \
  -d $DRIVELM_ROOT/data
unzip -q $DRIVELM_ROOT/downloads/drivelm_nus_imgs_val.zip \
  -d $DRIVELM_ROOT/data
mkdir -p $DRIVELM_ROOT/data/QA_dataset_nus
cp $DRIVELM_ROOT/downloads/v1_1_*_nus*.json \
  $DRIVELM_ROOT/data/QA_dataset_nus/
```

解压后必须能看到 `data/nuscenes/samples/CAM_FRONT` 等六个相机目录。`v1.0-mini.tgz` 中的雷达、激光雷达和 metadata 属于后续融合实验，不是本轮纯视觉复现的前置条件。

## 4. 生成数据

```bash
python reproduction/qwen_vl/build_dataset.py \
  --train-json $DRIVELM_ROOT/data/QA_dataset_nus/v1_1_train_nus.json \
  --val-json $DRIVELM_ROOT/data/QA_dataset_nus/v1_1_val_nus_q_only.json \
  --output-dir $DRIVELM_ROOT/reproduction/qwen_vl \
  --seed 42 --dev-ratio 0.1
```

每行保存场景、帧、任务、六相机绝对路径和消息。相机顺序固定为 front、front-left、front-right、back、back-left、back-right。train/dev 按场景哈希拆分，同一场景不会泄漏到两边。

## 5. 六图输入检查

```bash
python reproduction/qwen_vl/train.py \
  --model-path $DRIVELM_ROOT/models/Qwen2.5-VL-3B-Instruct \
  --train-jsonl $DRIVELM_ROOT/reproduction/qwen_vl/qwen_train.jsonl \
  --output-dir /tmp/drivelm-dry-run --max-train-samples 2 --dry-run
```

实测单样本 `input_ids=[1,824]`、`image_grid_thw=[6,3]`，模型收到六个独立视觉输入；助手监督 token 为 163，问题和图像 prompt 均被正确屏蔽。

## 6. 完整训练

```bash
CUDA_VISIBLE_DEVICES=1 python reproduction/qwen_vl/train.py \
  --model-path $DRIVELM_ROOT/models/Qwen2.5-VL-3B-Instruct \
  --train-jsonl $DRIVELM_ROOT/reproduction/qwen_vl/qwen_train.jsonl \
  --output-dir $DRIVELM_ROOT/models/qwen2.5-vl-3b-drivelm-sixcam \
  --device cuda:0 --batch-size 4 --gradient-accumulation-steps 1 \
  --epochs 1 --max-steps 0 --learning-rate 2e-4 --save-steps 500
```

默认每张图最多约 96 个视觉单元，控制六图显存。LoRA 可训练参数 18,576,384，占 0.4923%。真实冒烟训练已完成 1 次优化更新，loss 1.6296、用时 1.54 秒；batch=4 也已实测通过，单步 3.28 秒，全量一轮预计约 5 小时。每 500 步保存一次 adapter checkpoint。冒烟结果证明前向、反向与权重保存成立，但不代表模型收敛。

全量训练已完成 6,524 步。前 3,000 步后从 LoRA checkpoint 恢复，续训 3,524 步耗时 15,108.78 秒，续训阶段平均 loss 为 0.341855。最终 adapter 位于 `$DRIVELM_ROOT/models/qwen2.5-vl-3b-drivelm-sixcam`。

### 断点续训与独立后台会话

服务器的 PyTorch 默认设备顺序可能和 `nvidia-smi` 不同，因此显式设置 `CUDA_DEVICE_ORDER=PCI_BUS_ID`。例如从第 3,000 步恢复到总计 6,524 步：

```bash
tmux new-session -d -s drivelm_qwen_resume \
  "env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python \
  reproduction/qwen_vl/train.py \
  --model-path $DRIVELM_ROOT/models/Qwen2.5-VL-3B-Instruct \
  --train-jsonl $DRIVELM_ROOT/reproduction/qwen_vl/qwen_train.jsonl \
  --output-dir $DRIVELM_ROOT/models/qwen2.5-vl-3b-drivelm-sixcam \
  --device cuda:0 --batch-size 4 --max-steps 6524 --save-steps 500 \
  --resume-adapter $DRIVELM_ROOT/models/qwen2.5-vl-3b-drivelm-sixcam/checkpoint-3000 \
  --initial-step 3000"
```

用 `tmux attach -t drivelm_qwen_resume` 查看实时输出，按 `Ctrl-B` 后按 `D` 退出但不停止训练。checkpoint 保存 LoRA 权重，但当前版本没有保存 Adam 优化器状态，因此恢复时模型参数连续、优化器动量重新初始化。

## 7. 推理与本地评估

```bash
CUDA_VISIBLE_DEVICES=1 python reproduction/qwen_vl/infer.py \
  --model-path $DRIVELM_ROOT/models/Qwen2.5-VL-3B-Instruct \
  --adapter-path $DRIVELM_ROOT/models/qwen2.5-vl-3b-drivelm-sixcam \
  --input-jsonl $DRIVELM_ROOT/reproduction/qwen_vl/qwen_dev.jsonl \
  --output-json $DRIVELM_ROOT/reproduction/qwen_vl/dev_predictions.json \
  --device cuda:0 --max-new-tokens 256 --resume

python reproduction/qwen_vl/evaluate_offline.py \
  --references-jsonl $DRIVELM_ROOT/reproduction/qwen_vl/qwen_dev.jsonl \
  --predictions-json $DRIVELM_ROOT/reproduction/qwen_vl/dev_predictions.json \
  --output-json $DRIVELM_ROOT/reproduction/qwen_vl/dev_metrics.json
```

评估输出覆盖率、Exact Match、token-F1、ROUGE-L、各任务指标和多选准确率。它不是隐藏服务器分数。1-step 冒烟仅生成 4 条，token-F1 为 0.133，只用于验证链路。

推理会把每个完成样本立即写入同名 `.partial.jsonl`；使用 `--resume` 时自动跳过已经生成的 ID，全部完成后再按输入顺序写出官方 JSON 数组。

## 8. 官方 val 输出

把上面输入换为 `qwen_val_questions.jsonl`，输出换为 `output.json` 即可。文件严格采用官方 `[{'id': ..., 'answer': ...}]` 结构，多选回答规范化为一个 A/B/C/D 字母。val 没有公开真值，只能上传挑战服务器评估。

执行 `bash reproduction/qwen_vl/run_reproduction.sh` 可以依次运行构建、全量训练、dev 推理和评估。当前是纯视觉基线；下一版本将用同一场景 ID 和评测协议增加 radar token，开展 camera-only、radar-only、camera+radar 消融，再接 CARLA 决策闭环。
