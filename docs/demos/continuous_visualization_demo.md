# RadarMind × DriveLM 连续行驶问答视频

> 版本：Visualization v2.0（2026-08-07）。本版本以连续时间线替代 v1.0 六页总览式视频；本文同时作为版本说明、实现原理和完整复现手册。

这不是逐页切换的演示文稿。视频始终播放同一段 nuScenes 前视行驶序列；到关键场景时冻结当前帧，依次弹出问题和模型回答，停留后收起气泡，再从原时间线继续行驶。

## 1. 当前成片

第二条雨后夜间场景及完整复现见 [scene-1094 版本文档](continuous_visualization_scene1094.md)。

仓库视频：[assets/video/RadarMind_DriveLM_continuous.mp4](../../assets/video/RadarMind_DriveLM_continuous.mp4)

服务器完整产物：

~~~text
$DRIVELM_ROOT/reproduction/visualizations/continuous_scene_0061/
├── scene_0061_qwen.jsonl
├── scene_0061_predictions.json
├── scene_0061_predictions.json.partial.jsonl
├── RadarMind_DriveLM_continuous.mp4
└── RadarMind_DriveLM_continuous.report.json
~~~

实测媒体信息：1920×1080、24 FPS、H.264、774 帧、32.25 秒、AAC 静音音轨。源序列为 224 张 CAM_FRONT 图像，覆盖 19.15 秒；三次推理暂停各 4.35 秒。

| 成片时间 | 状态 | 内容 |
|---|---|---|
| 0.00–1.55 s | 行驶 | 连续前视图像，标题叠加但画面不停 |
| 1.55–5.90 s | 暂停 | Perception：重要目标识别 |
| 5.90–10.85 s | 行驶 | 从原关键帧继续播放 |
| 10.85–15.20 s | 暂停 | Prediction：下一位置依次关注哪些对象 |
| 15.20–23.25 s | 行驶 | 连续城市道路画面 |
| 23.25–27.60 s | 暂停 | Planning：安全驾驶动作 |
| 27.60–32.25 s | 行驶 | 恢复并播放到场景结尾 |

## 2. 数据真实性与边界

演示场景是 DriveLM token cc8c0bf57f984915a77078b10eb33198，对应 nuScenes scene-0061。

- 连续画面来自 nuScenes mini 的 samples/CAM_FRONT 和 sweeps/CAM_FRONT，按 sample_data.timestamp 排序。
- 三个事件只在 sample_data.is_key_frame=true 时触发，避免 sweep 共享 sample_token 造成重复暂停。
- 彩色检测框来自 DriveLM v1.1 的 key_object_infos.2d_bbox，画面明确标注为 DriveLM ANNOTATION BOXES。它们不是当前 VLM 自己回归出来的框。
- 气泡内容来自最终 Qwen2.5-VL-3B LoRA adapter 对该场景 57 条六相机 QA 的真实生成结果，没有手写答案。
- scene-0061 属于本地 SFT training split，因此右上角始终标明 TRAIN-SPLIT QUALITATIVE DEMO。它只展示交互形式；正式泛化指标仍以隔离的 77-scene dev 结果为准。

## 3. 环境与输入

~~~bash
conda activate radargym-rl
cd /path/to/radarmind-drivelm
python -m pip install -r reproduction/qwen_vl/requirements.txt
ffmpeg -version
~~~

需要以下输入：

~~~text
$DRIVELM_ROOT/data/QA_dataset_nus/v1_1_train_nus.json
$DRIVELM_ROOT/data/nuscenes/v1.0-mini/{scene,sample,sample_data}.json
$DRIVELM_ROOT/data/nuscenes/{samples,sweeps}/CAM_FRONT/*.jpg
$DRIVELM_ROOT/reproduction/qwen_vl/qwen_train.jsonl
$DRIVELM_ROOT/models/qwen2.5-vl-3b-drivelm-sixcam/
$DRIVELM_ROOT/models/Qwen2.5-VL-3B-Instruct/
~~~

nuScenes mini metadata 和连续相机 sweep 来自官方 v1.0-mini.tgz。解压时让 v1.0-mini、samples 和 sweeps 共用同一个 nuScenes 根目录：

~~~bash
tar -xzf /path/to/v1.0-mini.tgz -C $DRIVELM_ROOT/data/nuscenes
~~~

## 4. 提取 scene-0061

如果还没有完整 Qwen JSONL，先执行 [六视角复现手册](../current/reproduction_qwen_vl.md) 的数据构建。然后提取该场景全部记录：

~~~bash
python reproduction/qwen_vl/extract_scene_jsonl.py \
  --input-jsonl $DRIVELM_ROOT/reproduction/qwen_vl/qwen_train.jsonl \
  --scene-token cc8c0bf57f984915a77078b10eb33198 \
  --output-jsonl $DRIVELM_ROOT/reproduction/visualizations/continuous_scene_0061/scene_0061_qwen.jsonl
~~~

预期输出为 8 个关键帧、57 条问答。每条记录保留六个同步相机路径；连续视频只用 CAM_FRONT 作为背景，模型推理仍接收六视角。

## 5. 运行真实模型推理

~~~bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
python reproduction/qwen_vl/infer.py \
  --model-path $DRIVELM_ROOT/models/Qwen2.5-VL-3B-Instruct \
  --adapter-path $DRIVELM_ROOT/models/qwen2.5-vl-3b-drivelm-sixcam \
  --input-jsonl $DRIVELM_ROOT/reproduction/visualizations/continuous_scene_0061/scene_0061_qwen.jsonl \
  --output-json $DRIVELM_ROOT/reproduction/visualizations/continuous_scene_0061/scene_0061_predictions.json \
  --device cuda:0 --batch-size 4 --max-new-tokens 256 --resume
~~~

完成标志是 done: 57, total: 57，最终 JSON 是 57 个 id/answer 对象组成的数组。.partial.jsonl 是逐条落盘的断点文件。

## 6. 渲染连续视频

默认路径已与当前服务器对齐：

~~~bash
python reproduction/qwen_vl/visualize_continuous_demo.py
~~~

显式参数命令：

~~~bash
python reproduction/qwen_vl/visualize_continuous_demo.py \
  --nuscenes-root $DRIVELM_ROOT/data/nuscenes \
  --annotation $DRIVELM_ROOT/data/QA_dataset_nus/v1_1_train_nus.json \
  --jsonl $DRIVELM_ROOT/reproduction/visualizations/continuous_scene_0061/scene_0061_qwen.jsonl \
  --predictions $DRIVELM_ROOT/reproduction/visualizations/continuous_scene_0061/scene_0061_predictions.json \
  --scene-token cc8c0bf57f984915a77078b10eb33198 \
  --output $DRIVELM_ROOT/reproduction/visualizations/continuous_scene_0061/RadarMind_DriveLM_continuous.mp4 \
  --fps 24 --event-hold 4.35
~~~

脚本把 Pillow 生成的 RGB 帧直接写入 FFmpeg stdin，不在磁盘堆积临时 PNG。正常行驶段按 nuScenes 累计时间戳分配输出帧；暂停段复用当前关键帧，依次完成检测框、问题和回答的淡入淡出。

各场景事件位于脚本顶部 SCENE_PRESETS；scene-0061 的默认事件为：

~~~text
e0845f... / qa 0 / PERCEPTION / source 1.55 s
1e3d79... / qa 2 / PREDICTION / source 6.50 s
378a3a... / qa 5 / PLANNING / source 14.55 s
~~~

每次渲染会同时生成 .report.json，记录源帧数、源/输出时长、事件问题、真实预测和来源。验收命令：

~~~bash
ffprobe -v error \
  -show_entries format=duration,size:stream=codec_name,width,height,r_frame_rate,nb_frames \
  -of json \
  $DRIVELM_ROOT/reproduction/visualizations/continuous_scene_0061/RadarMind_DriveLM_continuous.mp4
~~~

## 7. 下载到 Mac

~~~bash
scp USER@SERVER_IP:$DRIVELM_ROOT/reproduction/visualizations/continuous_scene_0061/RadarMind_DriveLM_continuous.mp4 .
open RadarMind_DriveLM_continuous.mp4
~~~

也可以从 GitHub 的 assets/video/RadarMind_DriveLM_continuous.mp4 下载。旧的 RadarMind_DriveLM_v1.mp4 是六页总览版，只保留为历史产物，不再作为项目首页主展示。
