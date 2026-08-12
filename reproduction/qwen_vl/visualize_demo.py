#!/usr/bin/env python3
"""Render a DriveLM-style video from real six-camera data and model predictions."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps


WIDTH, HEIGHT = 1920, 1080
CAMERAS = (
    "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT",
)
NAVY = "#07111f"
PANEL = "#111f33"
BLUE = "#2f6bff"
CYAN = "#21d4c2"
WHITE = "#f6f8fc"
MUTED = "#a9b8ce"
RED = "#ff4d5e"
ORANGE = "#ffb547"
FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, size)


def wrap_lines(draw: ImageDraw.ImageDraw, text: str, text_font: ImageFont.FreeTypeFont, width: int) -> list[str]:
    words = text.replace("\n", " ").split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if draw.textlength(candidate, font=text_font) <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def text_block(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    text_font: ImageFont.FreeTypeFont,
    color: str,
    max_width: int,
    max_lines: int,
    spacing: int = 8,
) -> int:
    lines = wrap_lines(draw, text, text_font, max_width)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        while lines[-1] and draw.textlength(lines[-1] + "…", font=text_font) > max_width:
            lines[-1] = lines[-1][:-1]
        lines[-1] += "…"
    x, y = xy
    line_height = text_font.size + spacing
    for line in lines:
        draw.text((x, y), line, font=text_font, fill=color)
        y += line_height
    return y


def rounded_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: str, outline: str | None = None) -> None:
    draw.rounded_rectangle(box, radius=24, fill=fill, outline=outline, width=2 if outline else 1)


def header(canvas: Image.Image, section: str, subtitle: str, step: str) -> None:
    draw = ImageDraw.Draw(canvas)
    draw.polygon([(0, 0), (610, 0), (540, 112), (0, 112)], fill=BLUE)
    draw.text((42, 24), section, font=font(50, True), fill=WHITE)
    draw.text((640, 30), subtitle, font=font(27), fill=MUTED)
    draw.rounded_rectangle((1690, 29, 1870, 82), radius=26, fill="#16345e")
    draw.text((1723, 42), step, font=font(22, True), fill=CYAN)


def resolve_images(frame: dict[str, Any], annotation_path: Path) -> dict[str, Path]:
    return {
        camera: (annotation_path.parent / raw).resolve()
        for camera, raw in frame["image_paths"].items()
    }


def objects_for_camera(frame: dict[str, Any], camera: str) -> list[tuple[str, dict[str, Any]]]:
    return [
        (object_id, info)
        for object_id, info in frame["key_object_infos"].items()
        if f",{camera}," in object_id
    ]


def image_panel(
    canvas: Image.Image,
    image_path: Path,
    box: tuple[int, int, int, int],
    label: str,
    objects: list[tuple[str, dict[str, Any]]] | None = None,
    dim: float = 1.0,
) -> None:
    x1, y1, x2, y2 = box
    source = Image.open(image_path).convert("RGB")
    source_width, source_height = source.size
    fitted = ImageOps.fit(source, (x2 - x1, y2 - y1), method=Image.Resampling.LANCZOS)
    if dim != 1.0:
        fitted = ImageEnhance.Brightness(fitted).enhance(dim)
    canvas.paste(fitted, (x1, y1))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(box, outline="#314765", width=2)
    draw.rounded_rectangle((x1 + 16, y1 + 14, x1 + 260, y1 + 54), radius=18, fill="#07111fd9")
    draw.text((x1 + 29, y1 + 23), label, font=font(18, True), fill=WHITE)
    if not objects:
        return
    scale_x = (x2 - x1) / source_width
    scale_y = (y2 - y1) / source_height
    for index, (object_id, info) in enumerate(objects):
        bx1, by1, bx2, by2 = info["2d_bbox"]
        rect = (
            int(x1 + bx1 * scale_x), int(y1 + by1 * scale_y),
            int(x1 + bx2 * scale_x), int(y1 + by2 * scale_y),
        )
        color = RED if index % 2 == 0 else ORANGE
        draw.rectangle(rect, outline=color, width=5)
        short_id = object_id.split(",")[0].lstrip("<")
        title = short_id + " · " + str(info["Visual_description"]).rstrip(".")
        tw = int(draw.textlength(title, font=font(18, True))) + 22
        label_box = (rect[0], max(y1, rect[1] - 34), min(x2, rect[0] + tw), rect[1])
        draw.rectangle(label_box, fill=color)
        draw.text((label_box[0] + 10, label_box[1] + 6), title, font=font(18, True), fill=NAVY)


def qa_map(records: list[dict[str, Any]], predictions: dict[str, str]) -> dict[int, dict[str, str]]:
    result = {}
    for record in records:
        question = next(item["content"] for item in record["messages"] if item["role"] == "user")
        reference = next(item["content"] for item in record["messages"] if item["role"] == "assistant")
        result[int(record["qa_index"])] = {
            "question": str(question),
            "reference": str(reference),
            "prediction": predictions[record["id"]],
            "task": record["task"],
        }
    return result


def option_text(question: str, answer: str) -> str:
    match = re.search(rf"(?:^|\s){re.escape(answer)}\.\s*(.*?)(?=\s+[A-D]\.\s|$)", question)
    return match.group(1).strip() if match else answer


def render_slides(
    frame: dict[str, Any],
    images: dict[str, Path],
    qa: dict[int, dict[str, str]],
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    slides: list[Image.Image] = []

    title = Image.new("RGB", (WIDTH, HEIGHT), NAVY)
    for index, camera in enumerate(CAMERAS):
        col, row = index % 3, index // 3
        image_panel(title, images[camera], (col * 640, row * 540, (col + 1) * 640, (row + 1) * 540), camera, dim=0.45)
    overlay = Image.new("RGBA", title.size, (3, 10, 20, 115))
    title = Image.alpha_composite(title.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(title)
    draw.rounded_rectangle((165, 276, 1755, 810), radius=45, fill="#07111fd9", outline=BLUE, width=3)
    draw.text((260, 355), "RadarMind × DriveLM", font=font(84, True), fill=WHITE)
    draw.text((265, 475), "Six-Camera Graph VQA for Autonomous Driving", font=font(39), fill=CYAN)
    draw.rectangle((265, 558, 1650, 560), fill="#2d4260")
    draw.text((265, 607), "REAL nuScenes FRAME  •  REAL MODEL PREDICTIONS  •  NO DEV LEAKAGE", font=font(25, True), fill=MUTED)
    draw.rounded_rectangle((265, 682, 630, 742), radius=30, fill=BLUE)
    draw.text((310, 698), "Qwen2.5-VL-3B + LoRA", font=font(22, True), fill=WHITE)
    slides.append(title)

    sensor = Image.new("RGB", (WIDTH, HEIGHT), NAVY)
    header(sensor, "SENSORS", "Synchronized surround-view context", "01 / 05")
    for index, camera in enumerate(CAMERAS):
        col, row = index % 3, index // 3
        x1, y1 = 42 + col * 626, 150 + row * 382
        image_panel(sensor, images[camera], (x1, y1, x1 + 586, y1 + 330), camera)
    draw = ImageDraw.Draw(sensor)
    draw.rounded_rectangle((702, 930, 1218, 1005), radius=34, fill="#102944", outline=CYAN, width=2)
    draw.ellipse((735, 951, 757, 973), fill=CYAN)
    draw.text((780, 946), "6 views encoded as independent visual tokens", font=font(22, True), fill=WHITE)
    slides.append(sensor)

    perception = Image.new("RGB", (WIDTH, HEIGHT), NAVY)
    header(perception, "PERCEPTION", "What matters around the ego vehicle?", "02 / 05")
    image_panel(perception, images["CAM_FRONT"], (40, 145, 940, 651), "CAM_FRONT · annotation bbox", objects_for_camera(frame, "CAM_FRONT"))
    image_panel(perception, images["CAM_BACK"], (980, 145, 1880, 651), "CAM_BACK · annotation bbox", objects_for_camera(frame, "CAM_BACK"))
    draw = ImageDraw.Draw(perception)
    rounded_panel(draw, (40, 688, 690, 1025), "#14243a", BLUE)
    draw.text((74, 720), "DRIVER QUESTION", font=font(22, True), fill=CYAN)
    text_block(draw, (74, 765), qa[0]["question"], font(26, True), WHITE, 582, 6, 10)
    rounded_panel(draw, (728, 688, 1880, 1025), "#eff5ff")
    draw.rounded_rectangle((760, 716, 960, 765), radius=22, fill=BLUE)
    draw.text((793, 728), "MODEL OUTPUT", font=font(19, True), fill=WHITE)
    text_block(draw, (760, 794), qa[0]["prediction"], font(25), "#13223a", 1080, 7, 9)
    slides.append(perception)

    prediction = Image.new("RGB", (WIDTH, HEIGHT), NAVY)
    header(prediction, "PREDICTION", "Ordered object reasoning and future attention", "03 / 05")
    image_panel(prediction, images["CAM_FRONT"], (40, 150, 1105, 749), "CAM_FRONT", objects_for_camera(frame, "CAM_FRONT"), dim=0.88)
    image_panel(prediction, images["CAM_BACK"], (72, 510, 462, 730), "CAM_BACK", objects_for_camera(frame, "CAM_BACK"))
    draw = ImageDraw.Draw(prediction)
    rounded_panel(draw, (1145, 150, 1880, 749), "#13233a", "#29466d")
    draw.text((1182, 184), "REASONING TRAJECTORY", font=font(24, True), fill=CYAN)
    cards = [
        ("1", "Observe the front bus", "Object is going ahead"),
        ("2", "Check the nearby sedan", "Maintain a safe longitudinal gap"),
        ("3", "Verify the rear bus", "Continue at the same speed"),
    ]
    for index, (number, title_text, body) in enumerate(cards):
        y = 250 + index * 150
        draw.ellipse((1180, y, 1244, y + 64), fill=BLUE)
        draw.text((1201, y + 14), number, font=font(25, True), fill=WHITE)
        if index < len(cards) - 1:
            draw.line((1212, y + 68, 1212, y + 143), fill="#496689", width=4)
        draw.text((1270, y + 3), title_text, font=font(25, True), fill=WHITE)
        draw.text((1270, y + 45), body, font=font(21), fill=MUTED)
    rounded_panel(draw, (40, 785, 1880, 1028), "#eef5ff")
    draw.text((75, 815), "ACTUAL MODEL ANSWER", font=font(21, True), fill=BLUE)
    text_block(draw, (75, 858), qa[2]["prediction"], font(24), "#102039", 1730, 5, 8)
    slides.append(prediction)

    planning = Image.new("RGB", (WIDTH, HEIGHT), NAVY)
    image_panel(planning, images["CAM_FRONT"], (0, 0, WIDTH, HEIGHT), "CAM_FRONT", objects_for_camera(frame, "CAM_FRONT"), dim=0.36)
    overlay = Image.new("RGBA", planning.size, (4, 12, 24, 105))
    planning = Image.alpha_composite(planning.convert("RGBA"), overlay).convert("RGB")
    header(planning, "PLANNING", "From scene understanding to safe action", "04 / 05")
    draw = ImageDraw.Draw(planning)
    planning_cards = [
        ("CANDIDATE ACTION", qa[4]["prediction"], BLUE),
        ("COLLISION CHECK", qa[5]["prediction"], RED),
        ("SAFE ACTIONS", qa[6]["prediction"], CYAN),
        ("EGO BEHAVIOR", option_text(qa[7]["question"], qa[7]["prediction"]), ORANGE),
    ]
    for index, (label, answer, color) in enumerate(planning_cards):
        col, row = index % 2, index // 2
        x1, y1 = 90 + col * 900, 200 + row * 350
        x2, y2 = x1 + 840, y1 + 290
        draw.rounded_rectangle((x1, y1, x2, y2), radius=30, fill="#07111fe8", outline=color, width=3)
        draw.rounded_rectangle((x1 + 28, y1 + 28, x1 + 300, y1 + 78), radius=22, fill=color)
        draw.text((x1 + 50, y1 + 40), label, font=font(19, True), fill=NAVY if color != BLUE else WHITE)
        text_block(draw, (x1 + 34, y1 + 112), answer, font(29, True), WHITE, 770, 4, 12)
    slides.append(planning)

    result = Image.new("RGB", (WIDTH, HEIGHT), NAVY)
    header(result, "RESULT", "DriveLM-nuScenes v1 reproduction", "05 / 05")
    draw = ImageDraw.Draw(result)
    draw.text((90, 170), "A reproducible perception-to-planning VLM baseline", font=font(48, True), fill=WHITE)
    metrics = [
        ("0.7333", "DEV TOKEN-F1"),
        ("0.7159", "DEV ROUGE-L"),
        ("81.46%", "MULTI-CHOICE ACC."),
        ("15,480", "OFFICIAL VAL ANSWERS"),
    ]
    for index, (value, label) in enumerate(metrics):
        x1 = 90 + index * 450
        draw.rounded_rectangle((x1, 290, x1 + 405, 500), radius=28, fill=PANEL, outline="#28466d", width=2)
        draw.text((x1 + 35, 326), value, font=font(54, True), fill=CYAN if index < 3 else ORANGE)
        draw.text((x1 + 35, 420), label, font=font(19, True), fill=MUTED)
    stages = [
        ("6 CAMERAS", BLUE), ("VISUAL TOKENS", BLUE), ("QWEN-VL + LoRA", CYAN),
        ("GRAPH VQA", CYAN), ("SAFE ACTION", ORANGE),
    ]
    x = 90
    for index, (label, color) in enumerate(stages):
        draw.rounded_rectangle((x, 620, x + 300, 720), radius=30, fill="#13233a", outline=color, width=3)
        tw = draw.textlength(label, font=font(21, True))
        draw.text((x + (300 - tw) / 2, 655), label, font=font(21, True), fill=WHITE)
        if index < len(stages) - 1:
            draw.line((x + 310, 670, x + 355, 670), fill=MUTED, width=4)
            draw.polygon([(x + 355, 660), (x + 375, 670), (x + 355, 680)], fill=MUTED)
        x += 365
    draw.text((90, 845), "RadarMind", font=font(70, True), fill=WHITE)
    draw.text((555, 870), "Camera now · Radar fusion next", font=font(30, True), fill=CYAN)
    draw.text((90, 960), "Real data • Real predictions • Reproducible code", font=font(25), fill=MUTED)
    slides.append(result)

    paths = []
    for index, slide in enumerate(slides, start=1):
        path = output_dir / f"slide_{index:02d}.png"
        slide.save(path, quality=95)
        paths.append(path)
    return paths


def encode_video(slides: list[Path], output_path: Path, durations: list[float], fps: int) -> None:
    segment_dir = output_path.parent / "segments"
    segment_dir.mkdir(parents=True, exist_ok=True)
    segments = []
    for index, (slide, duration) in enumerate(zip(slides, durations, strict=True), start=1):
        segment = segment_dir / f"segment_{index:02d}.mp4"
        fade_out = max(duration - 0.45, 0)
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-loop", "1", "-i", str(slide), "-t", str(duration),
                "-vf", f"scale={WIDTH}:{HEIGHT},fade=t=in:st=0:d=0.4,fade=t=out:st={fade_out}:d=0.4,format=yuv420p",
                "-r", str(fps), "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-pix_fmt", "yuv420p", str(segment),
            ],
            check=True,
        )
        segments.append(segment)
    concat_file = segment_dir / "concat.txt"
    concat_file.write_text("".join("file " + str(path.resolve()) + "\n" for path in segments), encoding="utf-8")
    video_only = segment_dir / "video_only.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
         "-i", str(concat_file), "-c", "copy", str(video_only)],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(video_only), "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
            "-shortest", "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
            str(output_path),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-json", required=True, type=Path)
    parser.add_argument("--dev-jsonl", required=True, type=Path)
    parser.add_argument("--predictions-json", required=True, type=Path)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--frame-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    raw = json.loads(args.annotation_json.read_text(encoding="utf-8"))
    frame = raw[args.scene_id]["key_frames"][args.frame_id]
    records = []
    with args.dev_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["scene_id"] == args.scene_id and record["frame_id"] == args.frame_id:
                records.append(record)
    if not records:
        raise ValueError("The requested scene/frame is not present in the dev JSONL")
    predictions = {
        item["id"]: item["answer"]
        for item in json.loads(args.predictions_json.read_text(encoding="utf-8"))
    }
    missing = [record["id"] for record in records if record["id"] not in predictions]
    if missing:
        raise KeyError(f"Missing predictions for {missing[:3]}")
    images = resolve_images(frame, args.annotation_json)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    slides = render_slides(frame, images, qa_map(records, predictions), args.output_dir / "slides")
    output_path = args.output_dir / "radarmind_drivelm_demo.mp4"
    encode_video(slides, output_path, [4.0, 4.5, 7.0, 7.0, 7.0, 5.5], args.fps)
    report = {
        "scene_id": args.scene_id,
        "frame_id": args.frame_id,
        "slides": [str(path) for path in slides],
        "video": str(output_path),
        "duration_sec": 35.0,
        "resolution": [WIDTH, HEIGHT],
        "fps": args.fps,
        "bbox_source": "DriveLM key_object_infos.2d_bbox",
        "answer_source": str(args.predictions_json),
    }
    (args.output_dir / "visualization_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
