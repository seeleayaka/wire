from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


COLORS = (
    (238, 86, 74),
    (38, 166, 154),
    (255, 183, 77),
    (66, 133, 244),
    (171, 71, 188),
    (0, 137, 123),
    (244, 143, 177),
    (124, 179, 66),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local SAM3 cable mask probe.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-state-cache", type=Path)
    parser.add_argument("--prompt", default="cable")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    return parser.parse_args()


def render_overlay(
    image: Image.Image,
    masks: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
) -> Image.Image:
    base = np.asarray(image, dtype=np.float32).copy()
    for index, mask in enumerate(masks):
        color = np.asarray(COLORS[index % len(COLORS)], dtype=np.float32)
        active = mask.astype(bool)
        base[active] = base[active] * 0.55 + color * 0.45

    overlay = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    for index, (box, score) in enumerate(zip(boxes, scores)):
        color = COLORS[index % len(COLORS)]
        x0, y0, x1, y1 = (float(value) for value in box)
        draw.rectangle((x0, y0, x1, y1), outline=color, width=3)
        label = f"{index + 1}: {score:.3f}"
        text_box = draw.textbbox((x0, y0), label, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        label_y = max(0.0, y0 - text_height - 6)
        draw.rectangle(
            (x0, label_y, x0 + text_width + 8, label_y + text_height + 6),
            fill=color,
        )
        draw.text((x0 + 4, label_y + 3), label, fill=(255, 255, 255), font=font)
    return overlay


def render_all_masks_overlay(image: Image.Image, masks: np.ndarray) -> Image.Image:
    base = np.asarray(image, dtype=np.float32).copy()
    for index, mask in enumerate(masks):
        color = np.asarray(COLORS[index % len(COLORS)], dtype=np.float32)
        active = mask.astype(bool)
        base[active] = base[active] * 0.58 + color * 0.42
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def main() -> int:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, args.threads))

    timings: dict[str, float] = {}
    started = time.perf_counter()
    model = build_sam3_image_model(
        device="cpu",
        checkpoint_path=str(args.checkpoint),
        load_from_HF=False,
        enable_inst_interactivity=False,
        compile=False,
    )
    timings["model_build_seconds"] = time.perf_counter() - started

    processor = Sam3Processor(
        model,
        resolution=1008,
        device="cpu",
        confidence_threshold=args.threshold,
    )
    image = Image.open(args.input).convert("RGB")
    state_cache = args.image_state_cache or (args.output_dir / "image_state.pt")

    if state_cache.is_file():
        state = torch.load(state_cache, map_location="cpu", weights_only=False)
        timings["image_encoder_seconds"] = 0.0
        image_state_cache_reused = True
    else:
        image_state_cache_reused = False
        with torch.amp.autocast("cpu", dtype=torch.bfloat16):
            started = time.perf_counter()
            state = processor.set_image(image)
            timings["image_encoder_seconds"] = time.perf_counter() - started
        torch.save(state, state_cache)

    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        started = time.perf_counter()
        output = processor.set_text_prompt(state=state, prompt=args.prompt)
        timings["text_and_mask_seconds"] = time.perf_counter() - started

    masks = output["masks"].squeeze(1).detach().cpu().numpy().astype(bool)
    boxes = output["boxes"].detach().float().cpu().numpy()
    scores = output["scores"].detach().float().cpu().numpy()

    overlay = render_overlay(image, masks, boxes, scores)
    overlay.save(args.output_dir / "overlay.jpg", quality=95)
    render_all_masks_overlay(image, masks).save(
        args.output_dir / "overlay_all_masks.jpg", quality=95
    )
    Image.fromarray(np.any(masks, axis=0).astype(np.uint8) * 255).save(
        args.output_dir / "mask_union.png"
    )
    image.save(args.output_dir / "input.jpg", quality=95)
    for index, mask in enumerate(masks, start=1):
        Image.fromarray(mask.astype(np.uint8) * 255).save(
            args.output_dir / f"mask_{index:03d}.png"
        )

    report = {
        "input": str(args.input.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "prompt": args.prompt,
        "confidence_threshold": args.threshold,
        "device": "cpu",
        "threads": torch.get_num_threads(),
        "image_state_cache": str(state_cache.resolve()),
        "image_state_cache_reused": image_state_cache_reused,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "instance_count": int(len(scores)),
        "scores": [float(value) for value in scores],
        "boxes_xyxy": [[float(value) for value in box] for box in boxes],
        "mask_pixel_counts": [int(mask.sum()) for mask in masks],
        "mask_union_pixel_count": int(np.any(masks, axis=0).sum()),
        "timings": {key: round(value, 3) for key, value in timings.items()},
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
