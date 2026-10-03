"""Prepare a small Labelme pilot from Mendeley class-2 cable candidates.

The source labels are boxes, not cable masks. Each source class-2 box is written
as a rectangle-shaped polygon and marked ``needs_review`` so a human can redraw
the actual visible cable boundary in Labelme.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


SOURCE_CLASS = 2
SPLIT_MAP = {"train01": "train", "val01": "val", "test01": "test"}


def read_class2_boxes(label_path: Path) -> list[tuple[float, float, float, float]]:
    boxes: list[tuple[float, float, float, float]] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"{label_path}:{line_number}: expected 5 YOLO values")
        class_id, center_x, center_y, width, height = map(float, values)
        if int(class_id) == SOURCE_CLASS:
            boxes.append((center_x, center_y, width, height))
    return boxes


def select_images(dataset_root: Path, per_split: int) -> list[tuple[str, Path, Path]]:
    selected: list[tuple[str, Path, Path]] = []
    for source_split, target_split in SPLIT_MAP.items():
        candidates: list[tuple[int, Path, Path]] = []
        image_root = dataset_root / "images" / source_split
        label_root = dataset_root / "labels" / source_split
        for image_path in sorted(image_root.glob("*.JPG")):
            label_path = label_root / f"{image_path.stem}.txt"
            if label_path.is_file():
                count = len(read_class2_boxes(label_path))
                if count:
                    candidates.append((count, image_path, label_path))
        # Spread the pilot across sparse and dense examples instead of taking
        # only the first few nearly identical files.
        candidates.sort(key=lambda item: (item[0], item[1].name))
        if len(candidates) <= per_split:
            chosen = candidates
        else:
            indexes = [round(i * (len(candidates) - 1) / (per_split - 1)) for i in range(per_split)]
            chosen = [candidates[index] for index in indexes]
        selected.extend((target_split, image, label) for _count, image, label in chosen)
    return selected


def make_labelme_json(image_path: Path, boxes: list[tuple[float, float, float, float]]) -> dict[str, object]:
    from PIL import Image

    with Image.open(image_path) as image:
        width, height = image.size
    shapes: list[dict[str, object]] = []
    for index, (center_x, center_y, box_width, box_height) in enumerate(boxes):
        left = max(0.0, (center_x - box_width / 2) * width)
        top = max(0.0, (center_y - box_height / 2) * height)
        right = min(float(width - 1), (center_x + box_width / 2) * width)
        bottom = min(float(height - 1), (center_y + box_height / 2) * height)
        shapes.append(
            {
                "label": "cable",
                "points": [[left, top], [right, top], [right, bottom], [left, bottom]],
                "group_id": index,
                "description": "Initial class-2 box; redraw visible cable boundary",
                "shape_type": "polygon",
                "flags": {"needs_review": True},
            }
        )
    return {
        "version": "5.8.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-split", type=int, default=7)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing directory: {args.output}")
    args.output.mkdir(parents=True)
    selected = select_images(args.dataset_root, max(1, args.per_split))
    manifest: list[dict[str, object]] = []
    for split, image_path, label_path in selected:
        split_dir = args.output / split
        split_dir.mkdir(parents=True, exist_ok=True)
        target_image = split_dir / image_path.name
        shutil.copy2(image_path, target_image)
        boxes = read_class2_boxes(label_path)
        target_json = split_dir / f"{image_path.stem}.json"
        target_json.write_text(json.dumps(make_labelme_json(target_image, boxes), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest.append({"split": split, "image": str(target_image), "json": str(target_json), "initial_shapes": len(boxes)})
    (args.output / "README.txt").write_text(
        "Labelme cable pilot\n"
        "1. Open each split folder with Labelme.\n"
        "2. Keep label name exactly: cable.\n"
        "3. Redraw every rectangle around the visible cable polygon; split separate cables.\n"
        "4. Do not label background, motherboard, connector housing, or hidden cable sections.\n"
        "5. Save each JSON, then run convert_mendeley_cable_labelme.py.\n",
        encoding="utf-8",
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "images": len(manifest), "initial_shapes": sum(int(item["initial_shapes"]) for item in manifest)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
