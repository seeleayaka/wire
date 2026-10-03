"""Build a fixed-Dell local-ROI dataset for a supplied source target class.

This is deliberately narrower than "cable inspection": it learns whether a
source target annotation is visible within one of the fixed locations of the
photographed Dell chassis.  Class 4 / empty jack is the default, while class 3
can be used for disconnected plug ends.  ROI locations are learnt from the
supplied training split only; validation and test labels never influence their
geometry.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


SOURCE_EMPTY_JACK_CLASS = 4
SPLIT_MAP = {"train01": "train", "val01": "val", "test01": "test"}
CLASS_NAMES = {0: "no_documented_empty_jack", 1: "visible_empty_jack"}


def read_source_boxes(label_path: Path, source_class: int) -> list[tuple[float, float, float, float]]:
    """Read normalised xywh boxes for one source class."""
    boxes: list[tuple[float, float, float, float]] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"{label_path}:{line_number}: expected 5 fields")
        class_id, cx, cy, width, height = map(float, values)
        if int(class_id) == source_class:
            boxes.append((cx, cy, width, height))
    return boxes


def read_empty_jack_boxes(label_path: Path) -> list[tuple[float, float, float, float]]:
    """Backward-compatible class-4 convenience wrapper."""
    return read_source_boxes(label_path, SOURCE_EMPTY_JACK_CLASS)


def connected_components(points: list[tuple[float, float]], radius: float) -> list[list[tuple[float, float]]]:
    """Cluster nearby points without requiring another ML dependency."""
    remaining = set(range(len(points)))
    groups: list[list[tuple[float, float]]] = []
    while remaining:
        seed = remaining.pop()
        component = [seed]
        queue = [seed]
        while queue:
            index = queue.pop()
            x, y = points[index]
            neighbours = [
                candidate
                for candidate in remaining
                if (points[candidate][0] - x) ** 2 + (points[candidate][1] - y) ** 2 <= radius**2
            ]
            for candidate in neighbours:
                remaining.remove(candidate)
                queue.append(candidate)
                component.append(candidate)
        groups.append([points[index] for index in component])
    return groups


def train_only_rois(dataset_root: Path, radius: float, padding: float, source_class: int = SOURCE_EMPTY_JACK_CLASS) -> list[dict[str, object]]:
    """Create fixed target ROIs from *train01* labels, with conservative padding."""
    boxes = [
        box
        for label_path in sorted((dataset_root / "labels" / "train01").glob("*.txt"))
        for box in read_source_boxes(label_path, source_class)
    ]
    if not boxes:
        raise ValueError(f"train01 has no class-{source_class} annotations")
    groups = connected_components([(box[0], box[1]) for box in boxes], radius)
    records: list[dict[str, object]] = []
    for group in sorted(groups, key=len, reverse=True):
        member_boxes = [box for box in boxes if (box[0], box[1]) in set(group)]
        left = max(0.0, min(box[0] - box[2] / 2 for box in member_boxes) - padding)
        top = max(0.0, min(box[1] - box[3] / 2 for box in member_boxes) - padding)
        right = min(1.0, max(box[0] + box[2] / 2 for box in member_boxes) + padding)
        bottom = min(1.0, max(box[1] + box[3] / 2 for box in member_boxes) + padding)
        records.append(
            {
                "candidate_id": f"jack_roi_{len(records) + 1:02d}",
                "train_annotations": len(member_boxes),
                "roi_normalized_xyxy": [round(left, 6), round(top, 6), round(right, 6), round(bottom, 6)],
            }
        )
    return records


def grid_rois(columns: int, rows: int) -> list[dict[str, object]]:
    """Return non-overlapping camera-fixed grid ROIs for moving target ends."""
    if columns < 1 or rows < 1:
        raise ValueError("grid columns and rows must both be positive")
    records: list[dict[str, object]] = []
    for row in range(rows):
        for column in range(columns):
            records.append(
                {
                    "candidate_id": f"grid_r{row + 1:02d}_c{column + 1:02d}",
                    "train_annotations": None,
                    "roi_normalized_xyxy": [column / columns, row / rows, (column + 1) / columns, (row + 1) / rows],
                }
            )
    return records


def roi_contains_source_target(roi: list[float], boxes: list[tuple[float, float, float, float]]) -> bool:
    """Use an annotation centre to assign a source target to a local ROI."""
    left, top, right, bottom = roi
    return any(left <= cx <= right and top <= cy <= bottom for cx, cy, _width, _height in boxes)


def roi_contains_empty_jack(roi: list[float], boxes: list[tuple[float, float, float, float]]) -> bool:
    """Backward-compatible class-4 semantic wrapper."""
    return roi_contains_source_target(roi, boxes)


def expanded_crop_bounds(roi: list[float], context_scale: float) -> tuple[float, float, float, float]:
    """Return clipped normalised crop bounds with visual context around an ROI."""
    left, top, right, bottom = roi
    if not 0 <= left < right <= 1 or not 0 <= top < bottom <= 1:
        raise ValueError(f"invalid ROI {roi}")
    if context_scale < 1:
        raise ValueError("context_scale must be >= 1")
    cx, cy = (left + right) / 2, (top + bottom) / 2
    half_width, half_height = (right - left) * context_scale / 2, (bottom - top) * context_scale / 2
    return max(0.0, cx - half_width), max(0.0, cy - half_height), min(1.0, cx + half_width), min(1.0, cy + half_height)


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read image {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(path.suffix or ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise ValueError(f"cannot encode {path}")
    encoded.tofile(str(path))


def crop_roi(image: np.ndarray, roi: list[float], context_scale: float, output_size: int) -> np.ndarray:
    left, top, right, bottom = expanded_crop_bounds(roi, context_scale)
    height, width = image.shape[:2]
    x1, y1 = int(left * width), int(top * height)
    x2, y2 = max(x1 + 1, int(right * width)), max(y1 + 1, int(bottom * height))
    return cv2.resize(image[y1:y2, x1:x2], (output_size, output_size), interpolation=cv2.INTER_AREA)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True, help="Original Mendeley dataset root")
    parser.add_argument("--output", type=Path, required=True, help="New, empty derived dataset directory")
    parser.add_argument("--radius", type=float, default=0.045)
    parser.add_argument("--padding", type=float, default=0.025)
    parser.add_argument("--context-scale", type=float, default=3.0)
    parser.add_argument("--output-size", type=int, default=256)
    parser.add_argument("--source-class", type=int, default=SOURCE_EMPTY_JACK_CLASS)
    parser.add_argument("--positive-name", default="visible_empty_jack")
    parser.add_argument("--negative-name", default="no_documented_empty_jack")
    parser.add_argument("--grid-columns", type=int, default=None, help="Use a non-overlapping fixed grid instead of label-clustered ROIs")
    parser.add_argument("--grid-rows", type=int, default=None)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if not (args.dataset_root / "images").is_dir() or not (args.dataset_root / "labels").is_dir():
        raise FileNotFoundError("dataset root must contain images/ and labels/")

    using_grid = args.grid_columns is not None or args.grid_rows is not None
    if using_grid and (args.grid_columns is None or args.grid_rows is None):
        raise ValueError("--grid-columns and --grid-rows must be supplied together")
    rois = grid_rois(args.grid_columns, args.grid_rows) if using_grid else train_only_rois(args.dataset_root, args.radius, args.padding, args.source_class)
    args.output.mkdir(parents=True)
    records: list[dict[str, object]] = []
    counts: dict[str, Counter[str]] = {split: Counter() for split in SPLIT_MAP.values()}
    for source_split, target_split in SPLIT_MAP.items():
        for image_path in sorted((args.dataset_root / "images" / source_split).glob("*.JPG")):
            image = read_image(image_path)
            boxes = read_source_boxes(args.dataset_root / "labels" / source_split / f"{image_path.stem}.txt", args.source_class)
            for roi_index, roi_record in enumerate(rois):
                roi = roi_record["roi_normalized_xyxy"]
                assert isinstance(roi, list)
                label = int(roi_contains_source_target(roi, boxes))
                crop = crop_roi(image, roi, args.context_scale, args.output_size)
                relative_path = Path("crops") / target_split / str(roi_record["candidate_id"]) / f"{image_path.stem}.jpg"
                destination = args.output / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                write_image(destination, crop)
                records.append(
                    {
                        "split": target_split,
                        "image": image_path.name,
                        "roi_id": roi_record["candidate_id"],
                        "roi_index": roi_index,
                        "crop": relative_path.as_posix(),
                        "label": label,
                    }
                )
                counts[target_split]["crops"] += 1
                counts[target_split][args.positive_name if label else args.negative_name] += 1

    recipe = {
        "purpose": f"fixed-Dell local {args.positive_name} POC only",
        "source_dataset": str(args.dataset_root),
        "source_label": {"class": args.source_class, "meaning": args.positive_name},
        "class_names": {0: args.negative_name, 1: args.positive_name},
        "roi_policy": (
            f"{args.grid_columns}x{args.grid_rows} camera-fixed non-overlapping grid; source labels do not affect ROI placement"
            if using_grid
            else f"ROI geometry learnt exclusively from train01 class-{args.source_class} centres; labels for val/test do not affect ROI placement"
        ),
        "candidate_rois": rois,
        "context_scale": args.context_scale,
        "output_size": args.output_size,
        "warning": f"class 0 means no supplied class-{args.source_class} annotation in this ROI, not verified electrical continuity or correct end-to-end routing",
        "counts": {split: dict(count) for split, count in counts.items()},
    }
    (args.output / "recipe.json").write_text(json.dumps(recipe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.output / "index.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "rois": len(rois), "counts": recipe["counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
