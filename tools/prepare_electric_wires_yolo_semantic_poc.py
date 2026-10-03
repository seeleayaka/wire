"""Convert paired binary cable masks into an isolated one-class YOLO-seg POC set.

Each connected foreground component becomes one polygon.  This is deliberately
semantic-style supervision, not a claim that touching wires become instances.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


def numeric_id(path: Path) -> int:
    return int(path.name.split("_", 1)[0])


def yolo_polygons(mask_path: Path) -> list[str]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"cannot read mask: {mask_path}")
    foreground = (mask >= 128).astype(np.uint8) * 255
    height, width = foreground.shape
    contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    labels: list[str] = []
    for contour in contours:
        if cv2.contourArea(contour) < 8.0:
            continue
        epsilon = max(1.0, 0.0015 * cv2.arcLength(contour, True))
        polygon = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(polygon) < 3:
            continue
        normalized = polygon.astype(np.float32)
        normalized[:, 0] /= width
        normalized[:, 1] /= height
        coordinates = " ".join(f"{value:.6f}" for point in normalized for value in point)
        labels.append(f"0 {coordinates}")
    if not labels:
        raise ValueError(f"no usable foreground polygon in {mask_path}")
    return labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--val-count", type=int, default=8)
    args = parser.parse_args()

    rgb_paths = sorted(args.source_dir.glob("*_rgb.png"), key=numeric_id)
    if len(rgb_paths) <= args.val_count:
        raise ValueError("need more source pairs than validation pairs")
    args.dataset_dir.mkdir(parents=True, exist_ok=True)
    manifests: list[dict] = []
    # Every fourth ID is held out first, spreading the tiny POC over the
    # archive's numeric range rather than taking a contiguous background run.
    validation_ids = {numeric_id(path) for path in rgb_paths[::4][: args.val_count]}
    for rgb_path in rgb_paths:
        sample_id = numeric_id(rgb_path)
        mask_path = args.source_dir / f"{sample_id}_mask.png"
        if not mask_path.is_file():
            raise FileNotFoundError(f"matching mask missing: {mask_path}")
        split = "val" if sample_id in validation_ids else "train"
        image_dir = args.dataset_dir / "images" / split
        label_dir = args.dataset_dir / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        destination_image = image_dir / f"{sample_id}.png"
        destination_label = label_dir / f"{sample_id}.txt"
        shutil.copy2(rgb_path, destination_image)
        polygons = yolo_polygons(mask_path)
        destination_label.write_text("\n".join(polygons) + "\n", encoding="utf-8")
        manifests.append({"id": sample_id, "split": split, "polygons": len(polygons)})

    yaml_path = args.dataset_dir / "electric_wires_semantic_poc.yaml"
    yaml_path.write_text(
        "path: " + args.dataset_dir.as_posix() + "\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: cable_foreground\n",
        encoding="utf-8",
    )
    (args.dataset_dir / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "source": str(args.source_dir),
                "label_type": "binary semantic masks converted to connected-component polygons",
                "limitation": "touching or crossing wires may remain a single foreground instance",
                "samples": manifests,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"dataset={args.dataset_dir}")
    print(f"train={sum(item['split'] == 'train' for item in manifests)} val={sum(item['split'] == 'val' for item in manifests)}")


if __name__ == "__main__":
    main()
