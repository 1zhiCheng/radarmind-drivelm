#!/usr/bin/env python3
"""Generate evidence-based figures for the DriveLM reproduction report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont, ImageOps


REPO = Path(__file__).resolve().parents[2]
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
BLUE = "#3777ff"
CYAN = "#20c7aa"
ORANGE = "#ff9d42"
RED = "#e65362"
NAVY = "#102039"
MUTED = "#70839d"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("/mnt/data/zzy/drivelm/reproduction/qwen_vl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "reports/drivelm_reproduction_v1/assets",
    )
    return parser.parse_args()


def setup_plotting() -> None:
    family = font_manager.FontProperties(fname=FONT_PATH).get_name()
    plt.rcParams.update({
        "font.family": family,
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "#f8fafc",
        "axes.edgecolor": "#d4dce8",
        "axes.labelcolor": NAVY,
        "xtick.color": "#42546b",
        "ytick.color": "#42546b",
        "text.color": NAVY,
    })


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def dataset_chart(report: dict, output: Path) -> None:
    tasks = ["perception", "prediction", "planning", "behavior"]
    labels = ["感知", "预测", "规划", "行为"]
    splits = [
        ("Train", report["train_task_counts"], BLUE),
        ("Dev", report["dev_task_counts"], CYAN),
        ("Official val", report["val_task_counts"], ORANGE),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for ax, (name, values, color) in zip(axes, splits):
        counts = [values[t] for t in tasks]
        bars = ax.bar(labels, counts, color=color, width=0.68)
        ax.set_title(f"{name}：{sum(counts):,} QA", fontsize=14, fontweight="bold")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        for bar, value in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,}",
                    ha="center", va="bottom", fontsize=9)
    axes[0].set_ylabel("问答数量")
    fig.suptitle("DriveLM 数据划分与任务组成", fontsize=18, fontweight="bold", y=1.04)
    save_figure(fig, output / "dataset_task_distribution.png")


def metric_chart(metrics: dict, output: Path) -> None:
    order = ["overall", "perception", "prediction", "planning", "behavior"]
    labels = ["Overall", "Perception", "Prediction", "Planning", "Behavior"]
    rows = [metrics["overall"]] + [metrics["by_task"][key] for key in order[1:]]
    names = ["Exact Match", "Token-F1", "ROUGE-L"]
    keys = ["exact_match", "token_f1", "rouge_l"]
    colors = [BLUE, CYAN, ORANGE]
    x = list(range(len(labels)))
    width = 0.24
    fig, ax = plt.subplots(figsize=(12.8, 5.2))
    for j, (name, key, color) in enumerate(zip(names, keys, colors)):
        values = [row[key] * 100 for row in rows]
        positions = [v + (j - 1) * width for v in x]
        bars = ax.bar(positions, values, width=width, label=name, color=color)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 1.0, f"{value:.1f}",
                    ha="center", va="bottom", fontsize=8, rotation=90)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 105)
    ax.set_ylabel("得分（%）")
    ax.set_title("3,355 条场景隔离开发集：分任务离线指标", fontsize=17, fontweight="bold")
    ax.legend(ncol=3, loc="upper center")
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)
    save_figure(fig, output / "metrics_by_task.png")


def family_chart(analysis: dict, output: Path) -> None:
    mapping = [
        ("binary_reasoning", "二元推理"),
        ("moving_status_mc", "运动状态"),
        ("candidate_actions", "候选动作"),
        ("important_objects", "重要对象"),
        ("notice_graph", "关注图"),
        ("safe_actions", "安全动作"),
        ("collision_reasoning", "碰撞推理"),
        ("behavior_mc", "自车行为"),
    ]
    values = [analysis["family_metrics"][key]["token_f1"] * 100 for key, _ in mapping]
    labels = [label for _, label in mapping]
    colors = [CYAN if value >= 80 else ORANGE if value >= 60 else RED for value in values]
    fig, ax = plt.subplots(figsize=(11.8, 5.2))
    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1])
    ax.set_xlim(0, 105)
    ax.set_xlabel("Token-F1（%）")
    ax.set_title("问题族诊断：规划与结构接地仍是主要瓶颈", fontsize=17, fontweight="bold")
    ax.grid(axis="x", alpha=0.22)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values[::-1]):
        ax.text(value + 1.0, bar.get_y() + bar.get_height() / 2, f"{value:.1f}",
                va="center", fontsize=10)
    save_figure(fig, output / "question_family_f1.png")


def loss_chart(log_path: Path, output: Path) -> None:
    rows = []
    with log_path.open(errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith('{"step"'):
                rows.append(json.loads(line))
    steps = [row["step"] for row in rows]
    losses = [row["loss_this_run"] for row in rows]
    fig, ax = plt.subplots(figsize=(11.8, 4.6))
    ax.plot(steps, losses, color=BLUE, linewidth=2.2)
    ax.scatter([steps[0], steps[-1]], [losses[0], losses[-1]],
               color=[ORANGE, CYAN], s=55, zorder=3)
    ax.annotate(f"{losses[0]:.4f}", (steps[0], losses[0]),
                xytext=(12, 10), textcoords="offset points")
    ax.annotate(f"{losses[-1]:.4f}", (steps[-1], losses[-1]),
                xytext=(-45, 12), textcoords="offset points")
    ax.axvline(3000, color=RED, linestyle="--", alpha=0.65,
               label="从 checkpoint-3000 恢复")
    ax.set_xlabel("全局更新步")
    ax.set_ylabel("续训阶段累计平均 loss")
    ax.set_title("LoRA 续训记录（注意：不是逐步瞬时 loss）", fontsize=17, fontweight="bold")
    ax.grid(alpha=0.22)
    ax.legend()
    save_figure(fig, output / "continuation_mean_loss.png")


def box(ax, xy, width, height, text, color, text_size=11) -> None:
    patch = FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=1.7, edgecolor=color, facecolor=color + "18",
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text,
            ha="center", va="center", fontsize=text_size, fontweight="bold")


def arrow(ax, start, end) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                mutation_scale=15, linewidth=1.8, color=MUTED))


def pipeline_diagram(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 3.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    items = [
        (0.02, "DriveLM v1.1\n标注 + 六相机图像", BLUE),
        (0.205, "官方 QA 抽取\nseed=42", CYAN),
        (0.39, "按 scene 哈希拆分\n619 train / 77 dev", ORANGE),
        (0.575, "Qwen2.5-VL-3B\nLoRA SFT", BLUE),
        (0.76, "Dev 贪心生成\n3,355 / 3,355", CYAN),
        (0.91, "EM / F1 / ROUGE-L\n错误族分析", RED),
    ]
    widths = [0.15, 0.15, 0.15, 0.15, 0.13, 0.08]
    for idx, ((x, text, color), width) in enumerate(zip(items, widths)):
        box(ax, (x, 0.34), width, 0.33, text, color, 10 if idx == 5 else 11)
        if idx < len(items) - 1:
            arrow(ax, (x + width, 0.505), (items[idx + 1][0] - 0.008, 0.505))
    ax.text(0.02, 0.88, "可复现评估流水线", fontsize=19, fontweight="bold")
    ax.text(0.02, 0.12, "Official val 仅生成 15,480 条提交答案；无公开标签，不参与本地分数计算。",
            fontsize=11, color=RED)
    save_figure(fig, output / "reproduction_pipeline.png")


def model_diagram(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 4.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    centers = [
        (0.02, 0.27, 0.16, 0.46, "6 × 同步 RGB\n每图 ≤ 75,264 px", BLUE),
        (0.235, 0.27, 0.17, 0.46, "Vision Encoder\n32 层 · hidden 1280\npatch 14 / merge 2", CYAN),
        (0.46, 0.27, 0.15, 0.46, "视觉投影与\n多模态 token\nout hidden 2048", ORANGE),
        (0.66, 0.20, 0.20, 0.60, "Qwen2.5 LM\n36 层 · hidden 2048\n16 attention heads\ncontext 128K", BLUE),
        (0.91, 0.32, 0.075, 0.36, "答案\ntokens", CYAN),
    ]
    for x, y, w, h, text, color in centers:
        box(ax, (x, y), w, h, text, color, 10.5)
    for left, right in zip(centers, centers[1:]):
        arrow(ax, (left[0] + left[2], 0.5), (right[0] - 0.01, 0.5))
    ax.add_patch(FancyBboxPatch(
        (0.675, 0.05), 0.17, 0.11,
        boxstyle="round,pad=0.015", edgecolor=RED, facecolor="#fff3f4", linewidth=1.5,
    ))
    ax.text(0.76, 0.105, "LoRA r=8, α=16, dropout=0.05\nAttention + MLP projections",
            ha="center", va="center", fontsize=9.5, color=RED)
    arrow(ax, (0.76, 0.16), (0.76, 0.22))
    ax.text(0.02, 0.9, "六视角 Qwen2.5-VL LoRA 基线", fontsize=19, fontweight="bold")
    ax.text(0.02, 0.08, "可训练参数 18,576,384 / 总参数 3,773,199,360 = 0.4923%",
            fontsize=11, color=MUTED)
    save_figure(fig, output / "model_architecture.png")


def pil_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT_PATH, size)


def six_camera_montage(dev_jsonl: Path, output: Path) -> None:
    record = json.loads(dev_jsonl.open().readline())
    order = [
        "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT",
    ]
    tile_w, tile_h = 720, 405
    canvas = Image.new("RGB", (tile_w * 3, tile_h * 2), "#07111f")
    for idx, camera in enumerate(order):
        image = Image.open(record["images"][camera]).convert("RGB")
        image = ImageOps.fit(image, (tile_w, tile_h), Image.Resampling.LANCZOS)
        x, y = (idx % 3) * tile_w, (idx // 3) * tile_h
        canvas.paste(image, (x, y))
        draw = ImageDraw.Draw(canvas)
        draw.rounded_rectangle((x + 14, y + 14, x + 265, y + 55), 16, fill="#07111fdd")
        draw.text((x + 28, y + 23), camera, font=pil_font(20, True), fill="white")
    canvas.save(output / "six_camera_input.jpg", quality=93)


def qualitative_montage(output: Path) -> None:
    sources = [
        (REPO / "assets/images/repo/radarmind_drivelm_continuous.jpg",
         "scene-0061 · 白天施工路口"),
        (REPO / "assets/images/repo/radarmind_drivelm_scene1094_continuous.jpg",
         "scene-1094 · 雨后夜间行人场景"),
    ]
    tile_w, tile_h = 1280, 720
    canvas = Image.new("RGB", (tile_w, tile_h * 2 + 90), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (path, label) in enumerate(sources):
        image = Image.open(path).convert("RGB")
        image = ImageOps.fit(image, (tile_w, tile_h), Image.Resampling.LANCZOS)
        y = idx * tile_h
        canvas.paste(image, (0, y))
        draw.rounded_rectangle((28, y + 28, 580, y + 82), 20, fill="#07111fdd")
        draw.text((50, y + 40), label, font=pil_font(25, True), fill="white")
    draw.text((30, tile_h * 2 + 24),
              "连续播放 → 关键帧冻结 → 问题 → 真实模型回答 → 恢复时间线",
              font=pil_font(27, True), fill=NAVY)
    canvas.save(output / "qualitative_continuous_demos.jpg", quality=92)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    setup_plotting()
    dataset_path = args.artifact_root / "dataset_report.json"
    metrics_path = args.artifact_root / "dev_metrics.json"
    analysis_path = args.artifact_root / "dev_error_analysis.json"
    log_path = args.artifact_root / "resume_gpu1_from_3000.log"
    dataset_chart(json.loads(dataset_path.read_text()), output)
    metric_chart(json.loads(metrics_path.read_text()), output)
    family_chart(json.loads(analysis_path.read_text()), output)
    loss_chart(log_path, output)
    pipeline_diagram(output)
    model_diagram(output)
    six_camera_montage(args.artifact_root / "qwen_dev.jsonl", output)
    qualitative_montage(output)
    sources = [dataset_path, metrics_path, analysis_path, log_path]
    manifest = {
        "sources": {str(path): sha256(path) for path in sources},
        "outputs": sorted(path.name for path in output.iterdir() if path.is_file()),
    }
    (output / "asset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
