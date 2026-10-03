"""Audit Mendeley wiring-fault labels before building a port-state model.

This does not train or modify the dataset.  It counts the opaque YOLO class IDs
by image fault prefix and renders labelled samples, so their visual meaning can
be checked before any class name is assumed.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


KINDS = ("normal", "damaged", "disconnected", "misrouted")
COLORS = {
    1: (56, 186, 255),
    2: (95, 216, 99),
    3: (255, 128, 77),
    4: (217, 102, 255),
}


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    suffix = path.suffix or ".jpg"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise ValueError(f"cannot encode {path}")
    encoded.tofile(str(path))


def read_boxes(path: Path, width: int, height: int) -> list[tuple[int, tuple[int, int, int, int]]]:
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return []
    boxes: list[tuple[int, tuple[int, int, int, int]]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"{path}:{number}: expected 5 YOLO values")
        class_id, cx, cy, box_width, box_height = map(float, values)
        left = max(0, int(round((cx - box_width / 2) * width)))
        top = max(0, int(round((cy - box_height / 2) * height)))
        right = min(width - 1, int(round((cx + box_width / 2) * width)))
        bottom = min(height - 1, int(round((cy + box_height / 2) * height)))
        boxes.append((int(class_id), (left, top, right, bottom)))
    return boxes


def draw_boxes(image: np.ndarray, kind: str, boxes: list[tuple[int, tuple[int, int, int, int]]]) -> np.ndarray:
    canvas = image.copy()
    scale = max(0.6, image.shape[1] / 1800.0)
    for class_id, (left, top, right, bottom) in boxes:
        color = COLORS.get(class_id, (255, 255, 255))
        thickness = max(2, int(round(2 * scale)))
        cv2.rectangle(canvas, (left, top), (right, bottom), color, thickness)
        cv2.putText(
            canvas,
            f"class {class_id}",
            (left, max(25, top - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
    cv2.rectangle(canvas, (0, 0), (520, 62), (0, 0, 0), -1)
    cv2.putText(canvas, kind, (20, 43), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
    return canvas


def make_contact_sheet(paths: list[Path], target: Path) -> None:
    tiles: list[np.ndarray] = []
    for path in paths:
        image = read_image(path)
        scale = 360 / image.shape[1]
        tile = cv2.resize(image, (360, int(round(image.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        cv2.putText(tile, path.name, (8, tile.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        tiles.append(tile)
    if not tiles:
        return
    columns = 3
    height = max(tile.shape[0] for tile in tiles)
    rows: list[np.ndarray] = []
    for start in range(0, len(tiles), columns):
        row = tiles[start : start + columns]
        while len(row) < columns:
            row.append(np.zeros_like(tiles[0]))
        padded = [cv2.copyMakeBorder(tile, 0, height - tile.shape[0], 0, 0, cv2.BORDER_CONSTANT) for tile in row]
        rows.append(np.hstack(padded))
    write_image(target, np.vstack(rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-kind", type=int, default=6)
    args = parser.parse_args()

    images_root = args.dataset_root / "images"
    labels_root = args.dataset_root / "labels"
    output = args.output
    samples_root = output / "labelled_samples"
    samples_root.mkdir(parents=True, exist_ok=True)

    class_counts: dict[str, Counter[int]] = defaultdict(Counter)
    image_counts: Counter[str] = Counter()
    samples: dict[str, list[Path]] = defaultdict(list)
    for split_dir in sorted(images_root.iterdir()):
        if not split_dir.is_dir():
            continue
        for image_path in sorted(split_dir.glob("*.JPG")):
            kind = image_path.stem.split("_", 1)[0]
            if kind not in KINDS:
                continue
            image_counts[kind] += 1
            label_path = labels_root / split_dir.name / f"{image_path.stem}.txt"
            image = read_image(image_path)
            boxes = read_boxes(label_path, image.shape[1], image.shape[0])
            class_counts[kind].update(class_id for class_id, _box in boxes)
            if len(samples[kind]) < args.samples_per_kind:
                rendered = draw_boxes(image, kind, boxes)
                target = samples_root / f"{split_dir.name}_{image_path.stem}.jpg"
                write_image(target, rendered)
                samples[kind].append(target)

    all_samples = [path for kind in KINDS for path in samples[kind]]
    make_contact_sheet(all_samples, output / "contact_sheet.jpg")
    report = {
        "purpose": "audit opaque source label IDs before port-state model design",
        "dataset_root": str(args.dataset_root),
        "image_counts": dict(image_counts),
        "class_counts_by_image_prefix": {kind: dict(sorted(counts.items())) for kind, counts in sorted(class_counts.items())},
        "sample_files": {kind: [str(path) for path in paths] for kind, paths in samples.items()},
        "warning": "class IDs are intentionally not assigned a semantic name by this audit; verify labelled samples first.",
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
