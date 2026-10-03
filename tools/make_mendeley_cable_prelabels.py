"""Create review-only Labelme polygons from a cable YOLO-seg teacher model."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
from PIL import Image
from ultralytics import YOLO


def make_json(image_name: str, width: int, height: int, polygons: list[list[list[float]]]) -> dict:
    return {
        "version": "5.8.1",
        "flags": {},
        "shapes": [
            {
                "label": "cable",
                "points": points,
                "group_id": None,
                "description": "YOLO teacher proposal; review and correct before training",
                "shape_type": "polygon",
                "flags": {"auto_generated": True},
            }
            for points in polygons
        ],
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--imgsz", type=int, default=960)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing directory: {args.output}")
    args.output.mkdir(parents=True)

    roots = [args.source_root / split for split in ("train01", "val01", "test01")]
    labeled = {"damaged_001", "damaged_034", "damaged_044", "disconnected_026", "misrouted_005", "misrouted_026"}
    images = sorted(p for root in roots for p in root.glob("*.JPG") if p.stem not in labeled)
    images = images[: max(0, args.limit)]
    model = YOLO(str(args.weights))
    records: list[dict[str, object]] = []
    predictions = model.predict(
        [str(path) for path in images],
        imgsz=args.imgsz,
        conf=args.conf,
        iou=0.5,
        max_det=100,
        device="cpu",
        retina_masks=True,
        stream=True,
        verbose=False,
    )
    for source_path, result in zip(images, predictions):
        target_image = args.output / source_path.name
        shutil.copy2(source_path, target_image)
        with Image.open(source_path) as image:
            width, height = image.size
        polygons: list[list[list[float]]] = []
        confidences: list[float] = []
        if result.masks is not None:
            for points, confidence in zip(result.masks.xy, result.boxes.conf.tolist()):
                if len(points) < 3:
                    continue
                polygons.append([[round(float(x), 2), round(float(y), 2)] for x, y in points])
                confidences.append(round(float(confidence), 4))
        json_path = args.output / f"{source_path.stem}.json"
        json_path.write_text(json.dumps(make_json(target_image.name, width, height, polygons), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        overlay = result.plot()
        cv2.imwrite(str(args.output / f"{source_path.stem}_overlay.jpg"), overlay)
        records.append({"source": str(source_path), "image": target_image.name, "json": json_path.name, "polygons": len(polygons), "confidences": confidences})

    (args.output / "README.txt").write_text(
        "These are review-only YOLO teacher proposals. They are not ground truth.\n"
        "Open the JSON/image pairs in Labelme, delete false masks, redraw incomplete masks, then save corrected files elsewhere.\n"
        f"Model confidence threshold: {args.conf}\n",
        encoding="utf-8",
    )
    (args.output / "manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "images": len(records), "polygons": sum(int(r["polygons"]) for r in records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
