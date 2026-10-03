"""Render a read-only reference/aligned-inspection comparison sheet for visual audit."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    ok, data = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError(f"Cannot encode image: {path}")
    data.tofile(str(path))


def parse_box(value: str) -> tuple[int, int, int, int]:
    parts = [int(part) for part in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("box must be left,top,right,bottom")
    return tuple(parts)  # type: ignore[return-value]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--aligned-inspection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--box", type=parse_box, action="append", default=[])
    parser.add_argument("--padding", type=int, default=36)
    args = parser.parse_args()
    reference = read_image(args.reference)
    inspection = read_image(args.aligned_inspection)
    if reference.shape != inspection.shape:
        raise RuntimeError("Images must share dimensions")
    height, width = reference.shape[:2]
    regions = args.box or [(0, 0, width, height)]
    panels: list[np.ndarray] = []
    for index, (left, top, right, bottom) in enumerate(regions, 1):
        x1, y1 = max(0, left - args.padding), max(0, top - args.padding)
        x2, y2 = min(width, right + args.padding), min(height, bottom + args.padding)
        ref = reference[y1:y2, x1:x2].copy()
        test = inspection[y1:y2, x1:x2].copy()
        for panel, label in ((ref, "reference"), (test, "aligned wrong6")):
            cv2.putText(panel, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(panel, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (15, 15, 15), 1, cv2.LINE_AA)
        cv2.rectangle(test, (left - x1, top - y1), (right - x1, bottom - y1), (0, 215, 255), 2)
        pair = np.hstack((ref, test))
        cv2.putText(pair, f"region {index}", (8, max(44, pair.shape[0] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(pair, f"region {index}", (8, max(44, pair.shape[0] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
        panels.append(pair)
    canvas_width = max(panel.shape[1] for panel in panels)
    padded = [cv2.copyMakeBorder(panel, 0, 0, 0, canvas_width - panel.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for panel in panels]
    write_image(args.output, np.vstack(padded))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
