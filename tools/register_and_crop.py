"""Register a machine photo to its reference image and crop configured connectors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def load_anchors(path: Path, order: list[str]) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = {anchor["id"]: anchor for anchor in payload["anchors"]}
    missing = [anchor_id for anchor_id in order if anchor_id not in values]
    if missing:
        raise ValueError(f"Missing anchors in {path}: {', '.join(missing)}")
    return np.float32([[values[anchor_id]["x"], values[anchor_id]["y"]] for anchor_id in order])


def sharpness(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def main() -> None:
    parser = argparse.ArgumentParser(description="Register a full-resolution machine image and crop connector ROIs.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True, help="Anchor labels for --image")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    root = args.config.parent.parent
    reference_path = root / config["reference_image"]
    reference_labels = root / config["reference_anchor_labels"]
    reference = cv2.imread(str(reference_path))
    image = cv2.imread(str(args.image))
    if reference is None or image is None:
        raise SystemExit("Reference image or inspection image cannot be opened.")

    order = config["anchor_order"]
    transform = cv2.getPerspectiveTransform(load_anchors(args.anchors, order), load_anchors(reference_labels, order))
    ref_height, ref_width = reference.shape[:2]
    registered = cv2.warpPerspective(image, transform, (ref_width, ref_height))

    args.output.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output / "registered.jpg"), registered)
    result = {
        "source_image": str(args.image),
        "reference_image": str(reference_path),
        "sharpness": round(sharpness(image), 2),
        "connectors": {},
        "decision": "manual_review",
    }
    for connector_id, roi in config["connector_rois"].items():
        left, top, right, bottom = (int(roi[key]) for key in ("left", "top", "right", "bottom"))
        crop = registered[top:bottom, left:right]
        if crop.size == 0:
            result["connectors"][connector_id] = {"status": "uncertain", "reason": "invalid ROI"}
            continue
        crop_path = args.output / f"{connector_id}.jpg"
        cv2.imwrite(str(crop_path), crop)
        result["connectors"][connector_id] = {"status": "uncertain", "confidence": 0.0, "crop": str(crop_path)}
    (args.output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
