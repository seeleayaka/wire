"""Migrate trusted Labelme polygons to near-duplicate images.

This is geometric transfer for fixed-camera image sequences, not model inference.
The source JSON is never modified; every target is written to a separate review
directory together with an ECC alignment score and an overlay image.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


def load_gray(path: Path, scale: float) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(small, (5, 5), 0)


def estimate_transform(source: Path, target: Path, scale: float) -> tuple[float, np.ndarray]:
    """Return ECC score and a full-resolution source-to-target homography."""
    source_gray = load_gray(source, scale)
    target_gray = load_gray(target, scale)
    warp = np.eye(3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 150, 1e-6)
    score, small_warp = cv2.findTransformECC(
        target_gray,
        source_gray,
        warp,
        cv2.MOTION_HOMOGRAPHY,
        criteria,
        None,
        5,
    )
    scale_matrix = np.diag([scale, scale, 1.0]).astype(np.float32)
    full_warp = np.linalg.inv(scale_matrix) @ small_warp @ scale_matrix
    return float(score), full_warp.astype(np.float32)


def transform_points(points: list[list[float]], warp: np.ndarray, width: int, height: int) -> list[list[float]]:
    source = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(source, warp).reshape(-1, 2)
    transformed[:, 0] = np.clip(transformed[:, 0], 0, width - 1)
    transformed[:, 1] = np.clip(transformed[:, 1], 0, height - 1)
    return [[round(float(x), 2), round(float(y), 2)] for x, y in transformed]


def make_overlay(image_path: Path, shapes: list[dict], output_path: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    overlay = image.copy()
    colour = (0, 190, 255)
    for index, shape in enumerate(shapes, start=1):
        points = np.asarray(shape["points"], dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(overlay, [points], colour)
        cv2.polylines(image, [points], True, colour, 3, cv2.LINE_AA)
        x, y = points[0, 0]
        cv2.putText(image, str(index), (int(x), int(max(24, y))), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2, cv2.LINE_AA)
    blended = cv2.addWeighted(image, 0.76, overlay, 0.24, 0)
    if not cv2.imwrite(str(output_path), blended):
        raise OSError(f"could not write {output_path}")


def migrate_one(source_json: Path, target_image: Path, output: Path, scale: float, min_ecc: float) -> dict[str, object]:
    source_data = json.loads(source_json.read_text(encoding="utf-8"))
    # imagePath in a Labelme JSON is usually relative to the JSON directory.
    source_image = Path(str(source_data.get("imagePath", "")))
    if not source_image.is_file():
        source_image = source_json.parent / source_image
    if not source_image.is_file():
        raise FileNotFoundError(source_image)
    score, warp = estimate_transform(source_image, target_image, scale)
    record: dict[str, object] = {
        "target": target_image.name,
        "source_json": str(source_json),
        "ecc": round(score, 6),
        "accepted": score >= min_ecc,
    }
    if score < min_ecc:
        return record

    image = cv2.imread(str(target_image))
    if image is None:
        raise FileNotFoundError(target_image)
    height, width = image.shape[:2]
    shapes: list[dict] = []
    for shape in source_data.get("shapes", []):
        if shape.get("label") != "cable" or shape.get("shape_type") != "polygon":
            continue
        migrated = dict(shape)
        migrated["points"] = transform_points(shape["points"], warp, width, height)
        migrated["description"] = f"Migrated from {source_json.stem}; review boundary and cable movement. ECC={score:.6f}"
        migrated["flags"] = {**shape.get("flags", {}), "auto_migrated": True}
        shapes.append(migrated)

    target_json = output / f"{target_image.stem}.json"
    target_data = {
        "version": source_data.get("version", "5.8.1"),
        "flags": {},
        "shapes": shapes,
        "imagePath": target_image.name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }
    target_json.write_text(json.dumps(target_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(target_image, output / target_image.name)
    make_overlay(target_image, shapes, output / f"{target_image.stem}_overlay.jpg")
    record["polygons"] = len(shapes)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument("--source-image", type=Path, required=True)
    parser.add_argument("--targets", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=0.25)
    parser.add_argument("--min-ecc", type=float, default=0.98)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if not args.source_json.is_file() or not args.source_image.is_file():
        raise FileNotFoundError("source JSON/image is missing")
    args.output.mkdir(parents=True)
    records = []
    for target in args.targets:
        if not target.is_file():
            raise FileNotFoundError(target)
        records.append(migrate_one(args.source_json, target, args.output, args.scale, args.min_ecc))
    (args.output / "README.txt").write_text(
        "Review-only polygon migrations. Original annotations were not modified.\n"
        "Open accepted JSON/image pairs in Labelme; delete or adjust masks where cables moved.\n"
        f"Source: {args.source_json.name}; minimum ECC: {args.min_ecc}\n",
        encoding="utf-8",
    )
    (args.output / "manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "records": records}, ensure_ascii=False))


if __name__ == "__main__":
    main()
