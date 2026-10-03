from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the RT-DLO semantic cable stage.")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    return parser.parse_args()


def save_overlay(image_bgr: np.ndarray, mask: np.ndarray, path: Path) -> None:
    overlay = image_bgr.astype(np.float32).copy()
    color = np.zeros_like(overlay)
    color[:, :, 1] = 255
    active = mask.astype(bool)
    overlay[active] = overlay[active] * 0.45 + color[active] * 0.55
    cv2.imwrite(str(path), np.clip(overlay, 0, 255).astype(np.uint8))


def main() -> int:
    args = parse_args()
    for path in (args.repo, args.input, args.checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, args.threads))
    sys.path.insert(0, str(args.repo.resolve()))

    import model as network

    started = time.perf_counter()
    model = network.deeplabv3plus_resnet101(
        num_classes=1,
        output_stride=16,
        pretrained_backbone=False,
    )
    network.convert_to_separable_conv(model.classifier)
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    model_build_seconds = time.perf_counter() - started

    image_rgb = np.asarray(Image.open(args.input).convert("RGB"))
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    original_height, original_width = image_bgr.shape[:2]
    resized = cv2.resize(image_bgr, (args.width, args.height))
    tensor = torch.from_numpy(
        resized.transpose(2, 0, 1).astype(np.float32) / 255.0
    ).unsqueeze(0)

    started = time.perf_counter()
    with torch.inference_mode():
        probability = torch.sigmoid(model(tensor)).squeeze().cpu().numpy()
    inference_seconds = time.perf_counter() - started

    max_probability = float(probability.max())
    normalized = probability / max(max_probability, np.finfo(np.float32).eps)
    normalized_full = cv2.resize(
        normalized,
        (original_width, original_height),
        interpolation=cv2.INTER_LINEAR,
    )
    cv2.imwrite(
        str(args.output_dir / "probability_normalized.png"),
        np.clip(normalized_full * 255, 0, 255).astype(np.uint8),
    )
    cv2.imwrite(str(args.output_dir / "input.jpg"), image_bgr)

    coverage: dict[str, float] = {}
    for threshold in (0.3, 0.5, 0.7):
        mask = normalized_full >= threshold
        suffix = f"{threshold:.1f}".replace(".", "p")
        cv2.imwrite(
            str(args.output_dir / f"mask_{suffix}.png"),
            mask.astype(np.uint8) * 255,
        )
        save_overlay(image_bgr, mask, args.output_dir / f"overlay_{suffix}.jpg")
        coverage[f"threshold_{threshold:.1f}"] = round(float(mask.mean()), 6)

    report = {
        "input": str(args.input.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "input_size": [original_width, original_height],
        "model_size": [args.width, args.height],
        "threads": torch.get_num_threads(),
        "torch": torch.__version__,
        "strict_load_missing_keys": list(incompatible.missing_keys),
        "strict_load_unexpected_keys": list(incompatible.unexpected_keys),
        "max_probability_before_official_normalization": max_probability,
        "coverage": coverage,
        "timings": {
            "model_build_seconds": round(model_build_seconds, 3),
            "inference_seconds": round(inference_seconds, 3),
        },
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
