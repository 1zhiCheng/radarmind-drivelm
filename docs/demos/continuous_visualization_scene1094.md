# RadarMind × DriveLM scene-1094 连续视频

> 版本：Visualization v2.1（2026-08-07）。这是继 scene-0061 后的第二条独立连续场景视频，复用同一套六视角推理与暂停问答渲染流水线。

## 1. 成片与场景

仓库视频：[RadarMind_DriveLM_scene1094_continuous.mp4](../../assets/video/RadarMind_DriveLM_scene1094_continuous.mp4)

服务器产物：

~~~text
$DRIVELM_ROOT/reproduction/visualizations/continuous_scene_1094/
├── scene_1094_qwen.jsonl
├── scene_1094_predictions.json
├── scene_1094_predictions.json.partial.jsonl
├── RadarMind_DriveLM_scene1094_continuous.mp4
└── RadarMind_DriveLM_scene1094_continuous.report.json
~~~

场景信息：

- nuScenes 名称：scene-1094
- DriveLM scene token：de7d80a1f5fb4c3e82ce8a4f213b450a
- 场景描述：雨后夜间道路，多名行人、卡车、踏板车和横穿道路参与者
- CAM_FRONT 连续帧：232
- 源时间跨度：19.50 秒
- DriveLM 关键帧：8
- 六视角 QA：59
- 成片：1920×1080、24 FPS、782 帧、32.583 秒
- 视频 SHA-256：da4e683f4c29f3597ca0661a754a081a60fbaa375b08b39c4f6c51f5ec517bdb

scene-0757 虽然也是繁忙白天路口，但其 prediction 全是交通标志/路障二分类。为了保留未来关注顺序推理，本版本最终选择 scene-1094。

## 2. 连续时间线

| 成片时间 | 源时间 | 状态 | 内容 |
|---|---:|---|---|
| 0.00–5.50 s | 0.00–5.50 s | 行驶 | 雨后夜间路口连续前视 |
| 5.50–9.83 s | 5.50 s | 暂停 | Perception：过街行人和周围车辆 |
| 9.83–15.28 s | 5.50–10.95 s | 行驶 | 从冻结帧恢复原序列 |
| 15.28–19.62 s | 10.95 s | 暂停 | Prediction：下一位置依次关注三个对象 |
| 19.62–26.67 s | 10.95–18.00 s | 行驶 | 连续夜间道路 |
| 26.67–31.00 s | 18.00 s | 暂停 | Planning：安全动作 |
| 31.00–32.58 s | 18.00–19.50 s | 行驶 | 回到原时间线并播放至结尾 |

三个事件 preset：

~~~text
f65ffdc408fb4a0c8ef0d1614b47dce8 / qa 0 / PERCEPTION / source 5.50 s
6cb024831cce4b6e8acf85afb7cece6e / qa 2 / PREDICTION / source 10.95 s
fe40762a54e1414da73de751877ad576 / qa 6 / PLANNING / source 18.00 s
~~~

## 3. 真实模型输出

Perception 事件中，模型识别出前方行人、白色卡车、前左白色轿车以及其他视角对象。Prediction 事件中，模型给出三个对象的关注顺序：

1. 先关注前方行驶车辆并保持速度。
2. 再关注右前方静止对象并保持速度。
3. 最后关注后方行驶车辆并保持速度。

Planning 事件的实际生成结果为：

~~~text
Keep going at the same speed, decelerate gradually without braking.
~~~

所有气泡文本来自 scene_1094_predictions.json。彩色框来自 DriveLM key_object_infos.2d_bbox，并在画面中标明 DriveLM ANNOTATION BOXES，不将标注框冒充为 VLM 检测输出。该场景属于 SFT training split，画面右上角持续披露 TRAIN-SPLIT QUALITATIVE DEMO。

## 4. 提取 59 条场景数据

先按 [Qwen2.5-VL 六视角复现手册](../current/reproduction_qwen_vl.md) 生成完整 qwen_train.jsonl，然后执行：

~~~bash
cd /path/to/radarmind-drivelm
conda activate radargym-rl

python reproduction/qwen_vl/extract_scene_jsonl.py \
  --input-jsonl $DRIVELM_ROOT/reproduction/qwen_vl/qwen_train.jsonl \
  --scene-token de7d80a1f5fb4c3e82ce8a4f213b450a \
  --output-jsonl $DRIVELM_ROOT/reproduction/visualizations/continuous_scene_1094/scene_1094_qwen.jsonl
~~~

预期输出：

~~~json
{
  "records": 59,
  "frames": 8
}
~~~

## 5. 在 GPU 2 上运行六视角推理

~~~bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
python reproduction/qwen_vl/infer.py \
  --model-path $DRIVELM_ROOT/models/Qwen2.5-VL-3B-Instruct \
  --adapter-path $DRIVELM_ROOT/models/qwen2.5-vl-3b-drivelm-sixcam \
  --input-jsonl $DRIVELM_ROOT/reproduction/visualizations/continuous_scene_1094/scene_1094_qwen.jsonl \
  --output-json $DRIVELM_ROOT/reproduction/visualizations/continuous_scene_1094/scene_1094_predictions.json \
  --device cuda:0 --batch-size 4 --max-new-tokens 256 --resume
~~~

完成条件是 done=59、total=59。推理仍输入每个关键帧的六张同步相机图像；视频背景只选择连续 CAM_FRONT。

## 6. 渲染第二条视频

scene-1094 已加入 visualize_continuous_demo.py 的 SCENE_PRESETS。执行：

~~~bash
python reproduction/qwen_vl/visualize_continuous_demo.py \
  --nuscenes-root $DRIVELM_ROOT/data/nuscenes \
  --annotation $DRIVELM_ROOT/data/QA_dataset_nus/v1_1_train_nus.json \
  --jsonl $DRIVELM_ROOT/reproduction/visualizations/continuous_scene_1094/scene_1094_qwen.jsonl \
  --predictions $DRIVELM_ROOT/reproduction/visualizations/continuous_scene_1094/scene_1094_predictions.json \
  --scene-token de7d80a1f5fb4c3e82ce8a4f213b450a \
  --output $DRIVELM_ROOT/reproduction/visualizations/continuous_scene_1094/RadarMind_DriveLM_scene1094_continuous.mp4 \
  --fps 24 --event-hold 4.35
~~~

渲染报告自动写到同名 .report.json，其中记录 scene_name、232 张相机帧、三个事件、真实问题和预测来源。

## 7. 验收与下载

~~~bash
ffprobe -v error \
  -show_entries format=duration,size:stream=codec_name,width,height,r_frame_rate,nb_frames \
  -of json \
  $DRIVELM_ROOT/reproduction/visualizations/continuous_scene_1094/RadarMind_DriveLM_scene1094_continuous.mp4

ffmpeg -v error \
  -i $DRIVELM_ROOT/reproduction/visualizations/continuous_scene_1094/RadarMind_DriveLM_scene1094_continuous.mp4 \
  -f null -
~~~

Mac 下载：

~~~bash
scp USER@SERVER_IP:$DRIVELM_ROOT/reproduction/visualizations/continuous_scene_1094/RadarMind_DriveLM_scene1094_continuous.mp4 .
open RadarMind_DriveLM_scene1094_continuous.mp4
~~~
