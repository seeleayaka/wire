"""Run an isolated local wire-harness instance-segmentation POC.

This module stays outside the production review entry point. It converts a
local Ultralytics segmentation result (or the original Roboflow response) into
stable pixel masks and an auditable JSON report for later Vision Observation
and IntelliMan adapters. It does not produce an assembly verdict.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


MODEL_ID = "wire-harness-validation-model/1"
SERVERLESS_URL = "https://serverless.roboflow.com"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_MODEL = ROOT / "models" / "wire_harness_yolov8s_seg_best.pt"
TARGET_CLASSES = ("cable", "connector", "clip", "strap")
CLASS_COLORS = {
    "cable": (44, 180, 255),
    "connector": (255, 96, 72),
    "clip": (84, 210, 120),
    "strap": (198, 110, 230),
}


class RoboflowInferenceError(RuntimeError):
    """Raised when the remote POC request cannot produce JSON predictions."""


def read_image(path: str | Path) -> np.ndarray:
    """Read an image from a Unicode Windows path."""
    source = Path(path)
    image = cv2.imdecode(np.fromfile(str(source), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read image: {source}")
    return image


def write_image(path: str | Path, image: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    suffix = target.suffix.lower() or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise ValueError(f"cannot encode image: {target}")
    encoded.tofile(str(target))


def _canonical_class(value: Any) -> str:
    normalized = " ".join(str(value or "").strip().casefold().replace("_", " ").split())
    # The local model has two cable subclasses. Both feed the same cable mask,
    # while the original source class remains in the JSON evidence.
    if "cable" in normalized:
        return "cable"
    if "connector" in normalized:
        return "connector"
    if "clip" in normalized:
        return "clip"
    if "strap" in normalized:
        return "strap"
    return normalized


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _point_pairs(points: Any) -> list[list[float]]:
    if not isinstance(points, list):
        return []
    result: list[list[float]] = []
    for point in points:
        if isinstance(point, dict):
            x, y = _float_or_none(point.get("x")), _float_or_none(point.get("y"))
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            x, y = _float_or_none(point[0]), _float_or_none(point[1])
        else:
            continue
        if x is not None and y is not None:
            result.append([round(x, 3), round(y, 3)])
    return result


def _box_from_prediction(prediction: dict[str, Any], polygon: list[list[float]]) -> list[float] | None:
    center_x = _float_or_none(prediction.get("x"))
    center_y = _float_or_none(prediction.get("y"))
    width = _float_or_none(prediction.get("width"))
    height = _float_or_none(prediction.get("height"))
    if None not in (center_x, center_y, width, height):
        return [
            round(center_x - width / 2.0, 3),
            round(center_y - height / 2.0, 3),
            round(center_x + width / 2.0, 3),
            round(center_y + height / 2.0, 3),
        ]
    if len(polygon) >= 3:
        xs, ys = zip(*polygon)
        return [round(min(xs), 3), round(min(ys), 3), round(max(xs), 3), round(max(ys), 3)]
    return None


def normalize_predictions(raw: dict[str, Any], image_size: tuple[int, int]) -> tuple[list[dict[str, Any]], list[str]]:
    """Normalize the hosted response without discarding unknown project labels."""
    width, height = image_size
    predictions = raw.get("predictions", []) if isinstance(raw, dict) else []
    if not isinstance(predictions, list):
        return [], ["response_predictions_not_a_list"]
    records: list[dict[str, Any]] = []
    warnings: list[str] = []
    for index, item in enumerate(predictions):
        if not isinstance(item, dict):
            warnings.append(f"prediction_{index}_not_an_object")
            continue
        label = str(item.get("class", item.get("label", ""))).strip()
        class_name = _canonical_class(label)
        polygon = _point_pairs(item.get("points"))
        box = _box_from_prediction(item, polygon)
        if box is None:
            warnings.append(f"prediction_{index}_has_no_geometry")
            continue
        clipped_box = [
            round(max(0.0, min(float(width), box[0])), 3),
            round(max(0.0, min(float(height), box[1])), 3),
            round(max(0.0, min(float(width), box[2])), 3),
            round(max(0.0, min(float(height), box[3])), 3),
        ]
        clipped_polygon = [
            [round(max(0.0, min(float(width), point[0])), 3), round(max(0.0, min(float(height), point[1])), 3)]
            for point in polygon
        ]
        confidence = _float_or_none(item.get("confidence"))
        records.append(
            {
                "index": index,
                "class": class_name,
                "source_class": label,
                "confidence": round(max(0.0, min(1.0, confidence)), 6) if confidence is not None else None,
                "box_xyxy": clipped_box,
                "polygon": clipped_polygon,
                "geometry": "polygon" if len(clipped_polygon) >= 3 else "box_fallback",
            }
        )
    return records, warnings


def masks_from_predictions(records: Iterable[dict[str, Any]], image_size: tuple[int, int]) -> dict[str, np.ndarray]:
    """Build one binary mask per target class plus a union mask."""
    width, height = image_size
    masks = {name: np.zeros((height, width), dtype=np.uint8) for name in TARGET_CLASSES}
    masks["all_target"] = np.zeros((height, width), dtype=np.uint8)
    for record in records:
        class_name = record.get("class")
        if class_name not in TARGET_CLASSES:
            continue
        polygons = record.get("polygons")
        if not isinstance(polygons, list):
            polygons = [record.get("polygon", [])]
        valid_polygons = [
            np.asarray(polygon, dtype=np.float32).round().astype(np.int32).reshape(-1, 1, 2)
            for polygon in polygons
            if isinstance(polygon, list) and len(polygon) >= 3
        ]
        if valid_polygons:
            cv2.fillPoly(masks[class_name], valid_polygons, 255)
        else:
            left, top, right, bottom = (int(round(value)) for value in record["box_xyxy"])
            cv2.rectangle(masks[class_name], (left, top), (max(left, right - 1), max(top, bottom - 1)), 255, -1)
        cv2.bitwise_or(masks["all_target"], masks[class_name], dst=masks["all_target"])
    return masks


def render_overlay(image: np.ndarray, records: Iterable[dict[str, Any]]) -> np.ndarray:
    """Draw all predictions for human inspection, including non-target labels."""
    overlay = image.copy()
    for record in records:
        class_name = str(record.get("class", "unknown"))
        color = CLASS_COLORS.get(class_name, (170, 170, 170))
        polygons = record.get("polygons")
        if not isinstance(polygons, list):
            polygons = [record.get("polygon", [])]
        points_list = [
            np.asarray(polygon, dtype=np.float32).round().astype(np.int32).reshape(-1, 1, 2)
            for polygon in polygons
            if isinstance(polygon, list) and len(polygon) >= 3
        ]
        if points_list:
            fill = overlay.copy()
            cv2.fillPoly(fill, points_list, color)
            overlay = cv2.addWeighted(fill, 0.24, overlay, 0.76, 0)
            cv2.polylines(overlay, points_list, True, color, 2, cv2.LINE_AA)
        left, top, right, bottom = (int(round(value)) for value in record["box_xyxy"])
        cv2.rectangle(overlay, (left, top), (right, bottom), color, 2)
        confidence = record.get("confidence")
        suffix = f" {confidence:.2f}" if isinstance(confidence, (int, float)) else ""
        cv2.putText(overlay, f"{record.get('source_class') or class_name}{suffix}", (left, max(18, top - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    return overlay


def build_report(
    image_path: str | Path,
    image: np.ndarray,
    records: list[dict[str, Any]],
    warnings: list[str],
    *,
    confidence_threshold: float,
    model_id: str = MODEL_ID,
    source: str = "roboflow_serverless",
) -> dict[str, Any]:
    height, width = image.shape[:2]
    class_counts = {name: sum(record["class"] == name for record in records) for name in TARGET_CLASSES}
    target_records = [record for record in records if record["class"] in TARGET_CLASSES]
    return {
        "schema_version": 1,
        "poc": "wire_harness_instance_segmentation",
        "decision": "evidence_only_manual_review",
        "model_id": model_id,
        "source": source,
        "input": {
            "path": str(Path(image_path)),
            "sha256": hashlib.sha256(Path(image_path).read_bytes()).hexdigest(),
            "bytes": Path(image_path).stat().st_size,
            "image_size": [width, height],
        },
        "request": {
            "confidence_threshold": confidence_threshold,
            "mask_decode_mode": "accurate",
            "response_mask_format": "polygon",
        },
        "summary": {
            "prediction_count": len(records),
            "target_prediction_count": len(target_records),
            "target_class_counts": class_counts,
        },
        "predictions": records,
        "warnings": [
            "This POC supplies visual evidence only; it does not issue an assembly or NG verdict.",
            "Masks must be evaluated on target-device images before entering Vision Observation or IntelliMan.",
            *warnings,
        ],
    }


def infer_remote(
    image_path: str | Path,
    api_key: str,
    *,
    confidence_threshold: float = 0.25,
    timeout_seconds: float = 90.0,
    model_id: str = MODEL_ID,
) -> dict[str, Any]:
    """Upload one image to the explicitly selected Roboflow hosted model."""
    if not api_key.strip():
        raise ValueError("Roboflow API key is empty")
    try:
        import requests
    except ImportError as exc:
        raise RoboflowInferenceError("requests is required for remote POC inference") from exc
    source = Path(image_path)
    if source.stat().st_size > 20 * 1024 * 1024:
        raise ValueError("Roboflow Serverless Cloud API accepts files up to 20 MB")
    url = f"{SERVERLESS_URL.rstrip('/')}/{model_id.lstrip('/')}"
    params = {
        "api_key": api_key,
        "confidence": f"{confidence_threshold:.4f}",
        "format": "json",
        "mask_decode_mode": "accurate",
        "response_mask_format": "polygon",
        "disable_active_learning": "true",
    }
    with source.open("rb") as handle:
        response = requests.post(url, params=params, files={"file": (source.name, handle, "application/octet-stream")}, timeout=timeout_seconds)
    if not response.ok:
        detail = response.text[:1000].replace(api_key, "<redacted>")
        raise RoboflowInferenceError(f"Roboflow request failed ({response.status_code}): {detail}")
    try:
        payload = response.json()
    except ValueError as exc:
        raise RoboflowInferenceError("Roboflow returned a non-JSON response") from exc
    if not isinstance(payload, dict):
        raise RoboflowInferenceError("Roboflow returned an unexpected response shape")
    return payload


def _polygons_from_mask(mask: np.ndarray) -> list[list[list[float]]]:
    binary = np.where(mask > 0.5, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[list[list[float]]] = []
    for contour in contours:
        if len(contour) < 3:
            continue
        polygons.append([[round(float(x), 3), round(float(y), 3)] for x, y in contour.reshape(-1, 2)])
    return polygons


def infer_local(
    image_path: str | Path,
    model_path: str | Path,
    *,
    confidence_threshold: float = 0.25,
    image_size: int = 960,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Run a local YOLOv8 segmentation model and normalize its pixel masks."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is required for local segmentation inference") from exc

    source = Path(image_path)
    weights = Path(model_path)
    if not weights.is_file():
        raise FileNotFoundError(f"Local model weight is missing: {weights}")
    model = YOLO(str(weights))
    if model.task != "segment":
        raise ValueError(f"Expected an instance-segmentation model, got task={model.task!r}")
    result = model.predict(str(source), conf=confidence_threshold, imgsz=image_size, retina_masks=True, verbose=False)[0]
    if result.masks is None or result.boxes is None:
        return [], ["local_model_returned_no_instances"]

    image = read_image(source)
    height, width = image.shape[:2]
    mask_data = result.masks.data.detach().cpu().numpy()
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
    confidences = result.boxes.conf.detach().cpu().numpy()
    names = result.names
    records: list[dict[str, Any]] = []
    warnings: list[str] = []
    for index, (mask, box, class_id, confidence) in enumerate(zip(mask_data, boxes, class_ids, confidences)):
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        polygons = _polygons_from_mask(mask)
        if not polygons:
            warnings.append(f"local_prediction_{index}_has_empty_mask")
            continue
        source_class = str(names.get(int(class_id), class_id)) if isinstance(names, dict) else str(names[int(class_id)])
        records.append(
            {
                "index": index,
                "class": _canonical_class(source_class),
                "source_class": source_class,
                "confidence": round(float(confidence), 6),
                "box_xyxy": [round(float(value), 3) for value in box],
                "polygon": polygons[0],
                "polygons": polygons,
                "geometry": "mask_polygons",
            }
        )
    return records, warnings


def save_artifacts(output_dir: str | Path, image: np.ndarray, report: dict[str, Any]) -> dict[str, str]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    records = report["predictions"]
    masks = masks_from_predictions(records, (image.shape[1], image.shape[0]))
    paths: dict[str, str] = {}
    overlay_path = target / "overlay.jpg"
    write_image(overlay_path, render_overlay(image, records))
    paths["overlay"] = str(overlay_path)
    for class_name, mask in masks.items():
        mask_path = target / f"mask_{class_name}.png"
        write_image(mask_path, mask)
        paths[f"mask_{class_name}"] = str(mask_path)
    report_path = target / "report.json"
    report_with_artifacts = {**report, "artifacts": paths}
    report_path.write_text(json.dumps(report_with_artifacts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    paths["report"] = str(report_path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="One target-device image for this POC.")
    parser.add_argument("--output", type=Path, required=True, help="Directory for masks, overlay and report.")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--source", choices=("local", "roboflow"), default="local")
    parser.add_argument("--local-model", type=Path, default=DEFAULT_LOCAL_MODEL)
    parser.add_argument("--imgsz", type=int, default=960, help="Local YOLO inference image size.")
    parser.add_argument("--api-key-env", default="ROBOFLOW_API_KEY")
    parser.add_argument("--model-id", default=MODEL_ID)
    args = parser.parse_args()
    if not 0.0 <= args.confidence <= 1.0:
        raise SystemExit("--confidence must be between 0 and 1")
    image = read_image(args.image)
    if args.source == "local":
        records, warnings = infer_local(args.image, args.local_model, confidence_threshold=args.confidence, image_size=args.imgsz)
        model_id = str(args.local_model)
        report_source = "ultralytics_local"
    else:
        api_key = os.environ.get(args.api_key_env, "")
        if not api_key:
            raise SystemExit(f"Missing {args.api_key_env}; no image was uploaded.")
        raw = infer_remote(args.image, api_key, confidence_threshold=args.confidence, model_id=args.model_id)
        records, warnings = normalize_predictions(raw, (image.shape[1], image.shape[0]))
        model_id = args.model_id
        report_source = "roboflow_serverless"
    report = build_report(
        args.image,
        image,
        records,
        warnings,
        confidence_threshold=args.confidence,
        model_id=model_id,
        source=report_source,
    )
    paths = save_artifacts(args.output, image, report)
    print(json.dumps({"source": report_source, "model_id": model_id, "predictions": len(records), "artifacts": paths}, ensure_ascii=False))


if __name__ == "__main__":
    main()
