#!/usr/bin/env python3
"""Render a continuous DriveLM-style driving video with freeze-and-answer events.

The camera timeline always remains the visual background. At selected keyframes the
video freezes, annotation boxes appear, then a real model question/answer exchange
is animated on top before driving resumes.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageEnhance, ImageFont


WIDTH, HEIGHT = 1920, 1080
BLUE = "#3777ff"
CYAN = "#36dfc5"
WHITE = "#f7f9fd"
INK = "#102039"
MUTED = "#a9bad0"
ORANGE = "#ffb547"
RED = "#ff5968"
FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


@dataclass(frozen=True)
class CameraFrame:
    sample_token: str
    timestamp: int
    path: Path
    is_key_frame: bool


@dataclass(frozen=True)
class Event:
    sample_token: str
    qa_index: int
    section: str
    subtitle: str


SCENE_PRESETS = {
    "cc8c0bf57f984915a77078b10eb33198": (
        "scene-0061",
        (
            Event("e0845f5322254dafadbbed75aaa07969", 0, "PERCEPTION", "Important objects around the ego vehicle"),
            Event("1e3d79dae62742a0ad64c91679863358", 2, "PREDICTION", "What should the vehicle attend to next?"),
            Event("378a3a3e9af346308ab9dff8ced46d9c", 5, "PLANNING", "Choose a safe driving action"),
        ),
    ),
    "de7d80a1f5fb4c3e82ce8a4f213b450a": (
        "scene-1094",
        (
            Event("f65ffdc408fb4a0c8ef0d1614b47dce8", 0, "PERCEPTION", "Pedestrians and vehicles after rain"),
            Event("6cb024831cce4b6e8acf85afb7cece6e", 2, "PREDICTION", "What should the vehicle attend to next?"),
            Event("fe40762a54e1414da73de751877ad576", 6, "PLANNING", "Choose a safe driving action"),
        ),
    ),
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, size)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def ease(value: float) -> float:
    value = clamp(value)
    return 1.0 - (1.0 - value) ** 3


def wrap_lines(draw: ImageDraw.ImageDraw, text: str, text_font: ImageFont.FreeTypeFont, width: int) -> list[str]:
    words = re.sub(r"\s+", " ", text).strip().split(" ")
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


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    text_font: ImageFont.FreeTypeFont,
    fill: str,
    max_width: int,
    max_lines: int,
    spacing: int = 8,
) -> None:
    lines = wrap_lines(draw, text, text_font, max_width)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        while lines[-1] and draw.textlength(lines[-1] + "...", font=text_font) > max_width:
            lines[-1] = lines[-1][:-1]
        lines[-1] += "..."
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=text_font, fill=fill)
        y += text_font.size + spacing


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_records(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    result: dict[tuple[str, int], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            result[(row["frame_id"], int(row["qa_index"]))] = row
    return result


def load_predictions(path: Path) -> dict[str, str]:
    raw = load_json(path)
    if isinstance(raw, dict):
        return {str(key): str(value.get("answer", value) if isinstance(value, dict) else value) for key, value in raw.items()}
    return {str(row["id"]): str(row.get("answer", row.get("prediction", ""))) for row in raw}


def camera_timeline(nuscenes_root: Path, scene_token: str) -> list[CameraFrame]:
    metadata = nuscenes_root / "v1.0-mini"
    samples = {row["token"] for row in load_json(metadata / "sample.json") if row["scene_token"] == scene_token}
    frames = [
        CameraFrame(
            row["sample_token"],
            int(row["timestamp"]),
            nuscenes_root / row["filename"],
            bool(row["is_key_frame"]),
        )
        for row in load_json(metadata / "sample_data.json")
        if row["sample_token"] in samples and "/CAM_FRONT/" in row["filename"]
    ]
    frames.sort(key=lambda item: item.timestamp)
    missing = [str(item.path) for item in frames if not item.path.is_file()]
    if not frames or missing:
        raise FileNotFoundError(f"CAM_FRONT timeline is incomplete: {len(frames)} frames, {len(missing)} missing")
    return frames


def annotation_scene(annotation_path: Path, scene_token: str) -> dict[str, Any]:
    raw = load_json(annotation_path)
    if scene_token not in raw:
        raise KeyError(f"Scene {scene_token} is not in {annotation_path}")
    return raw[scene_token]


def fit_camera(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if image.size == (WIDTH, HEIGHT):
        return image
    source_ratio = image.width / image.height
    target_ratio = WIDTH / HEIGHT
    if source_ratio > target_ratio:
        crop_width = round(image.height * target_ratio)
        left = (image.width - crop_width) // 2
        image = image.crop((left, 0, left + crop_width, image.height))
    elif source_ratio < target_ratio:
        crop_height = round(image.width / target_ratio)
        top = (image.height - crop_height) // 2
        image = image.crop((0, top, image.width, top + crop_height))
    return image.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)


def objects_front(frame_annotation: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [
        (object_id, info)
        for object_id, info in frame_annotation.get("key_object_infos", {}).items()
        if ",CAM_FRONT," in object_id
    ]


def draw_hud(image: Image.Image, elapsed: float, total: float, scene_name: str, intro_alpha: float = 0.0) -> Image.Image:
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.rounded_rectangle((34, 30, 650, 100), radius=28, fill=(5, 15, 29, 205), outline=(56, 119, 255, 160), width=2)
    draw.ellipse((61, 54, 83, 76), fill=CYAN)
    draw.text((103, 47), "RadarMind × DriveLM", font=font(27, True), fill=WHITE)
    draw.text((394, 54), scene_name, font=font(18), fill=MUTED)
    draw.rounded_rectangle((1520, 31, 1885, 91), radius=26, fill=(5, 15, 29, 205))
    draw.text((1552, 51), "TRAIN-SPLIT QUALITATIVE DEMO", font=font(16, True), fill=ORANGE)
    bar_x1, bar_x2, bar_y = 40, 1880, 1042
    draw.rounded_rectangle((bar_x1, bar_y, bar_x2, bar_y + 8), radius=4, fill=(255, 255, 255, 70))
    progress = 0 if total <= 0 else clamp(elapsed / total)
    draw.rounded_rectangle((bar_x1, bar_y, bar_x1 + int((bar_x2 - bar_x1) * progress), bar_y + 8), radius=4, fill=CYAN)
    draw.text((40, 1000), f"{elapsed:04.1f}s", font=font(19, True), fill=WHITE)
    if intro_alpha > 0:
        alpha = int(230 * intro_alpha)
        draw.rounded_rectangle((116, 738, 1235, 960), radius=36, fill=(4, 13, 26, alpha), outline=(55, 119, 255, alpha), width=3)
        draw.text((164, 780), "CONTINUOUS MULTIMODAL DRIVING REASONING", font=font(25, True), fill=CYAN)
        draw.text((164, 832), "Drive · Pause · Reason · Resume", font=font(48, True), fill=WHITE)
        draw.text((166, 900), "Qwen2.5-VL-3B + DriveLM LoRA", font=font(24), fill=MUTED)
    return Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB")


def draw_annotation_boxes(layer: Image.Image, frame_annotation: dict[str, Any], opacity: float) -> None:
    draw = ImageDraw.Draw(layer)
    sx, sy = WIDTH / 1600.0, HEIGHT / 900.0
    colors = (RED, ORANGE, CYAN, BLUE)
    alpha = int(255 * opacity)
    for index, (object_id, info) in enumerate(objects_front(frame_annotation)):
        bbox = info.get("2d_bbox")
        if not bbox:
            continue
        x1, y1, x2, y2 = [int(value * scale) for value, scale in zip(bbox, (sx, sy, sx, sy))]
        color = colors[index % len(colors)]
        rgb = tuple(bytes.fromhex(color.lstrip("#")))
        draw.rectangle((x1, y1, x2, y2), outline=rgb + (alpha,), width=6)
        short_id = object_id.split(",")[0].lstrip("<")
        description = str(info.get("Visual_description", info.get("Category", "object"))).rstrip(".")
        label = f"{short_id} · {description}"
        label_font = font(20, True)
        label_width = int(draw.textlength(label, font=label_font)) + 24
        label_y = max(112, y1 - 38)
        draw.rounded_rectangle((x1, label_y, min(WIDTH - 20, x1 + label_width), label_y + 36), radius=9, fill=rgb + (alpha,))
        draw.text((x1 + 11, label_y + 6), label, font=label_font, fill=(6, 16, 30, alpha))
    draw.rounded_rectangle((40, 118, 398, 162), radius=18, fill=(5, 15, 29, int(210 * opacity)))
    draw.text((64, 129), "DriveLM ANNOTATION BOXES", font=font(17, True), fill=(255, 255, 255, alpha))


def qa_text(record: dict[str, Any], predictions: dict[str, str]) -> tuple[str, str]:
    question = next(str(item["content"]) for item in record["messages"] if item["role"] == "user")
    return question, predictions[record["id"]]


def render_event_frame(
    background: Image.Image,
    event: Event,
    frame_annotation: dict[str, Any],
    record: dict[str, Any],
    predictions: dict[str, str],
    scene_name: str,
    local_time: float,
    elapsed: float,
    total: float,
) -> Image.Image:
    image = ImageEnhance.Brightness(background).enhance(0.64)
    image = draw_hud(image, elapsed, total, scene_name)
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    box_alpha = ease(local_time / 0.38)
    if local_time > 3.9:
        box_alpha *= clamp((4.35 - local_time) / 0.45)
    draw_annotation_boxes(layer, frame_annotation, box_alpha)

    pulse = 0.5 + 0.5 * math.sin(local_time * 6.0)
    draw.rounded_rectangle((760, 31, 1128, 96), radius=28, fill=(255, 71, 89, 218), outline=(255, 255, 255, int(80 + 80 * pulse)), width=3)
    draw.rectangle((796, 51, 812, 76), fill=WHITE)
    draw.rectangle((822, 51, 838, 76), fill=WHITE)
    draw.text((862, 50), "PAUSED · REASONING", font=font(20, True), fill=WHITE)

    question, answer = qa_text(record, predictions)
    question_alpha = ease((local_time - 0.35) / 0.45)
    answer_alpha = ease((local_time - 1.20) / 0.50)
    fade = clamp((4.35 - local_time) / 0.45) if local_time > 3.9 else 1.0
    question_alpha *= fade
    answer_alpha *= fade

    if question_alpha > 0:
        qa = int(245 * question_alpha)
        slide = int(90 * (1.0 - ease((local_time - 0.35) / 0.45)))
        qbox = (890 + slide, 185, 1868 + slide, 430)
        draw.rounded_rectangle(qbox, radius=34, fill=(13, 35, 63, qa), outline=(55, 119, 255, qa), width=3)
        draw.polygon(((1818 + slide, 430), (1853 + slide, 430), (1841 + slide, 465)), fill=(13, 35, 63, qa))
        draw.rounded_rectangle((925 + slide, 212, 1158 + slide, 256), radius=20, fill=(55, 119, 255, qa))
        draw.text((951 + slide, 222), event.section, font=font(18, True), fill=(255, 255, 255, qa))
        draw.text((1190 + slide, 218), event.subtitle, font=font(18), fill=(174, 194, 218, qa))
        draw_wrapped(draw, (928 + slide, 282), question, font(27, True), (255, 255, 255, qa), 876, 4, 10)

    if answer_alpha > 0:
        aa = int(247 * answer_alpha)
        slide = int(110 * (1.0 - ease((local_time - 1.20) / 0.50)))
        abox = (82 - slide, 590, 1438 - slide, 968)
        draw.rounded_rectangle(abox, radius=38, fill=(242, 247, 255, aa), outline=(54, 223, 197, aa), width=4)
        draw.polygon(((112 - slide, 590), (168 - slide, 590), (123 - slide, 550)), fill=(242, 247, 255, aa))
        draw.ellipse((122 - slide, 630, 182 - slide, 690), fill=(54, 223, 197, aa))
        draw.text((141 - slide, 640), "AI", font=font(18, True), fill=(10, 35, 54, aa))
        draw.text((206 - slide, 628), "RadarMind answer", font=font(25, True), fill=(16, 32, 57, aa))
        draw.text((206 - slide, 668), "Actual Qwen2.5-VL + LoRA output", font=font(17), fill=(76, 98, 128, aa))
        draw.line((122 - slide, 718, 1395 - slide, 718), fill=(167, 185, 207, aa), width=2)
        answer_size = {"PERCEPTION": 22, "PREDICTION": 24, "PLANNING": 28}[event.section]
        answer_lines = {"PERCEPTION": 8, "PREDICTION": 7, "PLANNING": 6}[event.section]
        answer_font = font(answer_size)
        draw_wrapped(draw, (124 - slide, 746), answer, answer_font, (16, 32, 57, aa), 1265, answer_lines, 8)

    return Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nuscenes-root", type=Path, default=Path("/mnt/data/zzy/drivelm/data/nuscenes"))
    parser.add_argument("--annotation", type=Path, default=Path("/mnt/data/zzy/drivelm/data/QA_dataset_nus/v1_1_train_nus.json"))
    parser.add_argument("--jsonl", type=Path, default=Path("/mnt/data/zzy/drivelm/reproduction/visualizations/continuous_scene_0061/scene_0061_qwen.jsonl"))
    parser.add_argument("--predictions", type=Path, default=Path("/mnt/data/zzy/drivelm/reproduction/visualizations/continuous_scene_0061/scene_0061_predictions.json"))
    parser.add_argument("--output", type=Path, default=Path("/mnt/data/zzy/drivelm/reproduction/visualizations/continuous_scene_0061/RadarMind_DriveLM_continuous.mp4"))
    parser.add_argument("--scene-token", default="cc8c0bf57f984915a77078b10eb33198")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--event-hold", type=float, default=4.35)
    return parser.parse_args()


def encode(args: argparse.Namespace) -> dict[str, Any]:
    if args.scene_token not in SCENE_PRESETS:
        raise KeyError(f"No visualization preset for scene {args.scene_token}")
    scene_name, event_sequence = SCENE_PRESETS[args.scene_token]
    timeline = camera_timeline(args.nuscenes_root, args.scene_token)
    annotations = annotation_scene(args.annotation, args.scene_token)["key_frames"]
    records = load_records(args.jsonl)
    predictions = load_predictions(args.predictions)
    events = {event.sample_token: event for event in event_sequence}
    for event in event_sequence:
        if event.sample_token not in annotations or (event.sample_token, event.qa_index) not in records:
            raise KeyError(f"Missing event data for {event.sample_token} qa={event.qa_index}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    video_only = args.output.with_name(args.output.stem + ".video-only.mp4")
    ffmpeg = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}",
            "-r", str(args.fps), "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium",
            "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video_only),
        ],
        stdin=subprocess.PIPE,
    )
    if ffmpeg.stdin is None:
        raise RuntimeError("Could not open ffmpeg stdin")

    start = timeline[0].timestamp
    total = (timeline[-1].timestamp - start) / 1_000_000.0
    deltas = [(b.timestamp - a.timestamp) / 1_000_000.0 for a, b in zip(timeline, timeline[1:])]
    default_delta = sorted(deltas)[len(deltas) // 2]
    frame_count = 0
    event_report: list[dict[str, Any]] = []
    rendered_events: set[str] = set()
    for index, item in enumerate(timeline):
        elapsed = (item.timestamp - start) / 1_000_000.0
        next_elapsed = (
            (timeline[index + 1].timestamp - start) / 1_000_000.0
            if index + 1 < len(timeline) else elapsed + default_delta
        )
        repeats = max(1, round(next_elapsed * args.fps) - round(elapsed * args.fps))
        base = fit_camera(item.path)
        intro_alpha = clamp((1.65 - elapsed) / 0.55) if elapsed > 1.10 else ease(elapsed / 0.35)
        normal = draw_hud(base, elapsed, total, scene_name, intro_alpha)
        for _ in range(repeats):
            ffmpeg.stdin.write(normal.tobytes())
            frame_count += 1

        event = events.get(item.sample_token)
        if event is None or not item.is_key_frame or item.sample_token in rendered_events:
            continue
        rendered_events.add(item.sample_token)
        hold_frames = round(args.event_hold * args.fps)
        record = records[(event.sample_token, event.qa_index)]
        for hold_index in range(hold_frames):
            rendered = render_event_frame(
                base, event, annotations[event.sample_token], record, predictions, scene_name,
                hold_index / args.fps, elapsed, total,
            )
            ffmpeg.stdin.write(rendered.tobytes())
            frame_count += 1
        question, answer = qa_text(record, predictions)
        event_report.append({
            "time_seconds": round(elapsed, 3),
            "sample_token": event.sample_token,
            "section": event.section,
            "qa_index": event.qa_index,
            "question": question,
            "prediction": answer,
        })

    missing_events = set(events) - rendered_events
    if missing_events:
        ffmpeg.stdin.close()
        ffmpeg.wait()
        raise RuntimeError(f"Events were not found as CAM_FRONT keyframes: {sorted(missing_events)}")

    ffmpeg.stdin.close()
    return_code = ffmpeg.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, ffmpeg.args)

    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(video_only),
            "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
            "-shortest", "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart", str(args.output),
        ],
        check=True,
    )
    video_only.unlink()
    report = {
        "video": str(args.output),
        "scene_token": args.scene_token,
        "scene_name": scene_name,
        "split_disclosure": "DriveLM v1.1 training split qualitative demo",
        "camera_frames": len(timeline),
        "source_duration_seconds": round(total, 3),
        "output_frames": frame_count,
        "output_duration_seconds": round(frame_count / args.fps, 3),
        "fps": args.fps,
        "events": event_report,
        "provenance": {
            "camera": "nuScenes mini CAM_FRONT continuous sample_data",
            "boxes": "DriveLM v1.1 key_object_infos annotations",
            "answers": f"{args.predictions.name} from the trained Qwen2.5-VL-3B LoRA adapter",
        },
    }
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    report = encode(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
