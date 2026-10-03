"""Render raw cabinet reference-versus-inspection pairs for manual labelling.

This tool intentionally performs no registration, candidate detection, scoring,
or annotation drawing. It only puts original images side by side for a human
reviewer. Output remains outside the project by default.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np


IMAGE_ROOT = Path(r"C:\Users\HUAWEI\Desktop\案例\线材柜子\4")
DEFAULT_OUTPUT = Path(os.environ["LOCALAPPDATA"]) / "Temp" / "cabinet_manual_only_raw_pairs_20260826_v1"
NAMES = ["wrong.png", *[f"wrong{index}.png" for index in range(2, 14)]]


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def draw_manual_box(image: np.ndarray, box: list[int], label: str) -> None:
    left, top, right, bottom = (int(value) for value in box)
    green = (70, 220, 70)
    cv2.rectangle(image, (left, top), (right, bottom), green, 3, cv2.LINE_AA)
    cv2.rectangle(image, (left, max(0, top - 22)), (left + 55, top), green, -1, cv2.LINE_AA)
    cv2.putText(image, label, (left + 4, top - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)


def pair(reference: np.ndarray, inspection: np.ndarray, name: str, groups: list[dict]) -> np.ndarray:
    if reference.shape != inspection.shape:
        inspection = cv2.resize(inspection, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_AREA)
    inspection = inspection.copy()
    for group in groups:
        draw_manual_box(inspection, group["bbox"], group["id"])
    header = np.full((38, reference.shape[1] * 2, 3), (35, 35, 35), dtype=np.uint8)
    suffix = "manual boxes only" if groups else "no algorithm overlay"
    cv2.putText(header, f"{name} | left: raw reference | right: raw inspection | {suffix}", (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 2, cv2.LINE_AA)
    return np.vstack([header, np.hstack([reference, inspection])])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", type=Path, default=IMAGE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--annotations", type=Path, help="Optional manual-only standard JSON; green boxes are drawn on the right image.")
    args = parser.parse_args()
    reference = read_image(args.image_root / "right.png")
    args.output.mkdir(parents=True, exist_ok=True)
    annotations: dict[str, list[dict]] = {}
    if args.annotations:
        document = json.loads(args.annotations.read_text(encoding="utf-8"))
        annotations = {case["image"]: case.get("groups", []) for case in document["images"]}
    for name in NAMES:
        image = pair(reference, read_image(args.image_root / name), name, annotations.get(name, []))
        target = args.output / name.replace(".png", "_raw_pair.jpg")
        if not cv2.imwrite(str(target), image):
            raise RuntimeError(f"Cannot write {target}")
    print(args.output)


if __name__ == "__main__":
    main()
