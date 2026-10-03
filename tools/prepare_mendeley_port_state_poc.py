"""Prepare an isolated two-class port-state dataset from Mendeley source labels.

The source has no class map.  Visual audit of labelled source images establishes
the following mapping for this fixed Dell chassis only:
  3 -> unplugged_plug
  4 -> unplugged_jack

Source bounding boxes are encoded as four-corner polygons so a locally available
YOLO segmentation checkpoint can train a detector without downloading a new
detect-only base model.  The resulting masks are rectangles, not true masks.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path


SOURCE_TO_TARGET = {3: 0, 4: 1}
TARGET_NAMES = {0: "unplugged_plug", 1: "unplugged_jack"}
SPLIT_MAP = {"train01": "train", "val01": "val", "test01": "test"}


def convert_label_text(text: str) -> list[str]:
    """Keep only source classes 3/4 and convert their boxes to rectangle polygons."""
    converted: list[str] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"label line {number}: expected 5 values, got {len(values)}")
        source_class, center_x, center_y, width, height = map(float, values)
        target_class = SOURCE_TO_TARGET.get(int(source_class))
        if target_class is None:
            continue
        left = max(0.0, center_x - width / 2)
        top = max(0.0, center_y - height / 2)
        right = min(1.0, center_x + width / 2)
        bottom = min(1.0, center_y + height / 2)
        if right <= left or bottom <= top:
            raise ValueError(f"label line {number}: clipped box has no area")
        converted.append(
            f"{target_class} {left:.6f} {top:.6f} {right:.6f} {top:.6f} "
            f"{right:.6f} {bottom:.6f} {left:.6f} {bottom:.6f}"
        )
    return converted


def link_or_copy(source: Path, destination: Path) -> str:
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True, help=".../Predictive Maintenance for Electrical Wiring Faults")
    parser.add_argument("--output", type=Path, required=True, help="New, empty derived-dataset directory")
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing derived dataset: {args.output}")
    images_root = args.dataset_root / "images"
    labels_root = args.dataset_root / "labels"
    if not images_root.is_dir() or not labels_root.is_dir():
        raise FileNotFoundError("dataset root must contain images/ and labels/")

    args.output.mkdir(parents=True)
    counts: dict[str, Counter[str]] = {split: Counter() for split in SPLIT_MAP.values()}
    link_methods: Counter[str] = Counter()
    samples: list[dict[str, object]] = []
    for source_split, target_split in SPLIT_MAP.items():
        image_dir = images_root / source_split
        label_dir = labels_root / source_split
        target_images = args.output / "images" / target_split
        target_labels = args.output / "labels" / target_split
        target_images.mkdir(parents=True)
        target_labels.mkdir(parents=True)
        for source_image in sorted(image_dir.glob("*.JPG")):
            source_label = label_dir / f"{source_image.stem}.txt"
            target_image = target_images / source_image.name
            target_label = target_labels / source_label.name
            link_methods[link_or_copy(source_image, target_image)] += 1
            converted = convert_label_text(source_label.read_text(encoding="utf-8"))
            target_label.write_text("\n".join(converted) + ("\n" if converted else ""), encoding="utf-8")
            per_image = Counter(int(line.split()[0]) for line in converted)
            counts[target_split]["images"] += 1
            counts[target_split]["positive_images"] += int(bool(converted))
            counts[target_split]["unplugged_plug_boxes"] += per_image[0]
            counts[target_split]["unplugged_jack_boxes"] += per_image[1]
            samples.append({"split": target_split, "image": source_image.name, "labels": dict(per_image)})

    yaml_path = args.output / "mendeley_port_state_rectseg.yaml"
    yaml_path.write_text(
        "path: " + args.output.as_posix() + "\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: unplugged_plug\n"
        "  1: unplugged_jack\n",
        encoding="utf-8",
    )
    manifest = {
        "purpose": "fixed-Dell visible port-state POC only",
        "source_dataset": str(args.dataset_root),
        "source_class_mapping_verified_by_visual_audit": {str(key): TARGET_NAMES[value] for key, value in SOURCE_TO_TARGET.items()},
        "ignored_source_classes": {"1": "damaged_cable", "2": "misrouted_cable"},
        "label_encoding": "source bounding boxes converted to four-corner polygons; this is rectangular segmentation supervision, not cable masks",
        "split_policy": "retain supplied train01/val01/test01 divisions; not a claim of independent field validation",
        "link_methods": dict(link_methods),
        "counts": {split: dict(counter) for split, counter in counts.items()},
        "samples": samples,
    }
    (args.output / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": str(args.output), "counts": manifest["counts"], "yaml": str(yaml_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
