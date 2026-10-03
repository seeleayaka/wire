from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


# 参考图片中四个稳定机器锚点的像素坐标，按左上、右上、右下、左下顺序填写。
# 选机壳角、螺丝孔中心、铭牌角等刚性特征；不要选择线缆。
REFERENCE_ANCHORS = np.float32([
    [100, 100],
    [980, 100],
    [980, 2000],
    [100, 2000],
])
CURRENT_ANCHORS = np.float32([
    [100, 100],
    [980, 100],
    [980, 2000],
    [100, 2000],
])

CONNECTOR_ROIS = {
    "J1": (160, 260, 380, 480),
    "J2": (620, 260, 840, 480),
}


def check_image_quality(image: np.ndarray) -> tuple[bool, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return sharpness >= 50.0, sharpness


def register_to_reference(image: np.ndarray) -> np.ndarray:
    transform = cv2.getPerspectiveTransform(CURRENT_ANCHORS, REFERENCE_ANCHORS)
    height, width = image.shape[:2]
    return cv2.warpPerspective(image, transform, (width, height))


def classify_connector(connector_image: np.ndarray) -> tuple[str, float]:
    return "uncertain", 0.0


def inspect(image_path: Path, output_dir: Path) -> dict:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot open image: {image_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    quality_ok, sharpness = check_image_quality(image)
    result = {
        "image": str(image_path),
        "image_quality_ok": quality_ok,
        "sharpness": round(sharpness, 2),
        "connectors": {},
    }
    if not quality_ok:
        result["decision"] = "retake_photo"
        return result

    registered = register_to_reference(image)
    cv2.imwrite(str(output_dir / "registered.jpg"), registered)

    for connector_id, (left, top, right, bottom) in CONNECTOR_ROIS.items():
        crop = registered[top:bottom, left:right]
        if crop.size == 0:
            result["connectors"][connector_id] = {"status": "uncertain", "reason": "ROI outside image"}
            continue
        crop_path = output_dir / f"{connector_id}.jpg"
        cv2.imwrite(str(crop_path), crop)
        status, confidence = classify_connector(crop)
        result["connectors"][connector_id] = {
            "status": status,
            "confidence": confidence,
            "crop": str(crop_path),
        }

    result["decision"] = "manual_review"
    return result


if __name__ == "__main__":
    input_image = Path("data/test.jpg")
    report = inspect(input_image, Path("output"))
    Path("output/result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
