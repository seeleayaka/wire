"""PROTOTYPE ONLY — reference-image assembly inspection.

This is a small, local experiment inspired by three open-source capabilities:
1. feature/pattern matching for automatic image registration;
2. normal-only anomaly maps (a lightweight analogue of Anomalib workflows);
3. optional Ultralytics YOLO detections when a task-specific model is supplied.

It answers: given same-checkpoint, similar-view photos, can the system produce
a small set of connector or memory-slot regions for a human to review?
It never certifies an electrical connection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def read_image(path: Path) -> np.ndarray | None:
    """Unicode-safe Windows image read."""
    try:
        raw = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(raw, cv2.IMREAD_COLOR) if raw.size else None


def write_image(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f"Cannot encode image: {path}")
    encoded.tofile(str(path))


def align_to(reference: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Pattern-matching layer: align candidate to the verified reference view."""
    orb = cv2.ORB_create(nfeatures=6000, fastThreshold=7)
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    candidate_gray = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)
    ref_points, ref_descriptors = orb.detectAndCompute(ref_gray, None)
    candidate_points, candidate_descriptors = orb.detectAndCompute(candidate_gray, None)
    report: dict[str, Any] = {"reference_keypoints": len(ref_points), "candidate_keypoints": len(candidate_points), "matches": 0, "inliers": 0}
    if ref_descriptors is None or candidate_descriptors is None:
        report["reason"] = "no visual descriptors"
        return None, report
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(candidate_descriptors, ref_descriptors, k=2)
    matches = [first for first, second in pairs if first.distance < 0.72 * second.distance]
    report["matches"] = len(matches)
    if len(matches) < 12:
        report["reason"] = "too few feature matches"
        return None, report
    source = np.float32([candidate_points[item.queryIdx].pt for item in matches]).reshape(-1, 1, 2)
    destination = np.float32([ref_points[item.trainIdx].pt for item in matches]).reshape(-1, 1, 2)
    homography, inlier_mask = cv2.findHomography(source, destination, cv2.RANSAC, 5.0)
    if homography is None or inlier_mask is None:
        report["reason"] = "no geometric transform"
        return None, report
    report["inliers"] = int(inlier_mask.ravel().sum())
    report["inlier_ratio"] = round(report["inliers"] / len(matches), 4)
    if report["inliers"] < 10:
        report["reason"] = "insufficient geometric agreement"
        return None, report
    height, width = reference.shape[:2]
    return cv2.warpPerspective(candidate, homography, (width, height)), report


def read_checklist(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return [{"id": "whole_image", "kind": "generic", "roi": [0.0, 0.0, 1.0, 1.0]}]
    payload = json.loads(path.read_text(encoding="utf-8"))
    checks = payload.get("checks", [])
    if not checks:
        raise ValueError("Checklist needs a non-empty checks list.")
    for check in checks:
        if "id" not in check or "roi" not in check or len(check["roi"]) != 4:
            raise ValueError("Every check needs id and normalized roi [left, top, right, bottom].")
    return checks


def crop(image: np.ndarray, roi: list[float]) -> np.ndarray:
    height, width = image.shape[:2]
    left, top, right, bottom = roi
    x1, y1 = round(left * width), round(top * height)
    x2, y2 = round(right * width), round(bottom * height)
    return image[max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)]


def anomaly_map(normal_images: list[np.ndarray], inspection: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Normal-only anomaly layer; more reference photos reduce lighting noise."""
    normals = np.stack([cv2.cvtColor(item, cv2.COLOR_BGR2GRAY).astype(np.float32) for item in normal_images])
    target = cv2.cvtColor(inspection, cv2.COLOR_BGR2GRAY).astype(np.float32)
    baseline = np.median(normals, axis=0)
    if len(normals) == 1:
        score = np.abs(target - baseline)
        threshold = max(25.0, float(np.percentile(score, 99.0)))
    else:
        mad = np.median(np.abs(normals - baseline), axis=0)
        score = np.abs(target - baseline) / np.maximum(6.0, 1.4826 * mad)
        threshold = max(3.5, float(np.percentile(score, 99.0)))
    blurred = cv2.GaussianBlur(score, (5, 5), 0)
    mask = np.uint8(blurred >= threshold) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)
    return score, mask, threshold


def regions_from_mask(mask: np.ndarray, score: np.ndarray) -> list[dict[str, Any]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(80, int(mask.shape[0] * mask.shape[1] * 0.0001))
    regions = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        regions.append({"left": x, "top": y, "right": x + width, "bottom": y + height, "area": round(area, 1), "score": round(float(score[y : y + height, x : x + width].mean()), 3)})
    return sorted(regions, key=lambda item: float(item["score"]) * float(item["area"]), reverse=True)


def optional_yolo(image: np.ndarray, model_path: Path | None) -> dict[str, Any]:
    """YOLO integration point. A custom connector/RAM model is required for semantics."""
    if model_path is None:
        return {"configured": False, "reason": "No custom YOLO weight supplied."}
    try:
        from ultralytics import YOLO
    except ImportError:
        return {"configured": False, "reason": "ultralytics is not installed in this prototype environment."}
    model = YOLO(str(model_path))
    result = model.predict(image, verbose=False)[0]
    detections = []
    if result.boxes is not None:
        names = result.names
        for box in result.boxes:
            cls = int(box.cls[0])
            detections.append({"class": str(names[cls]), "confidence": round(float(box.conf[0]), 4), "xyxy": [round(float(value), 1) for value in box.xyxy[0].tolist()]})
    return {"configured": True, "detections": detections}


def main() -> None:
    parser = argparse.ArgumentParser(description="Reference-image assembly inspection prototype.")
    parser.add_argument("--reference", type=Path, action="append", required=True, help="Verified-good image; repeat for multiple normal references.")
    parser.add_argument("--inspection", type=Path, required=True, help="Production image of the same checkpoint and view.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checklist", type=Path, help="Optional per-check-item normalized ROI JSON.")
    parser.add_argument("--yolo-model", type=Path, help="Optional custom connector/RAM YOLO weight.")
    args = parser.parse_args()

    canonical = read_image(args.reference[0])
    target = read_image(args.inspection)
    if canonical is None or target is None:
        raise SystemExit("Cannot read reference or inspection image.")
    aligned_target, target_alignment = align_to(canonical, target)
    report: dict[str, Any] = {"prototype": True, "decision": "manual_review", "target_alignment": target_alignment, "checks": [], "yolo": {"configured": False}}
    args.output.mkdir(parents=True, exist_ok=True)
    if aligned_target is None:
        report["reason"] = "Retake the production photo with the same checkpoint and view."
        (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    aligned_normals = [canonical]
    reference_alignment = []
    for reference_path in args.reference[1:]:
        image = read_image(reference_path)
        if image is None:
            reference_alignment.append({"image": str(reference_path), "accepted": False, "reason": "unreadable"})
            continue
        aligned, metrics = align_to(canonical, image)
        reference_alignment.append({"image": str(reference_path), "accepted": aligned is not None, "alignment": metrics})
        if aligned is not None:
            aligned_normals.append(aligned)
    report["normal_reference_count"] = len(aligned_normals)
    report["reference_alignment"] = reference_alignment
    report["yolo"] = optional_yolo(aligned_target, args.yolo_model)
    write_image(args.output / "aligned_inspection.jpg", aligned_target)

    for check in read_checklist(args.checklist):
        target_crop = crop(aligned_target, check["roi"])
        normal_crops = [crop(image, check["roi"]) for image in aligned_normals]
        score, mask, threshold = anomaly_map(normal_crops, target_crop)
        regions = regions_from_mask(mask, score)
        annotated = target_crop.copy()
        for number, region in enumerate(regions, start=1):
            x1, y1, x2, y2 = (int(region[key]) for key in ("left", "top", "right", "bottom"))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(annotated, f"Review {number}", (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        check_dir = args.output / str(check["id"])
        check_dir.mkdir(exist_ok=True)
        write_image(check_dir / "inspection.jpg", target_crop)
        write_image(check_dir / "anomaly_mask.png", mask)
        write_image(check_dir / "review_regions.jpg", annotated)
        score_image = cv2.normalize(score, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        write_image(check_dir / "anomaly_heatmap.jpg", cv2.applyColorMap(score_image, cv2.COLORMAP_JET))
        report["checks"].append({"id": check["id"], "kind": check.get("kind", "generic"), "expected": check.get("expected", "visual_match"), "threshold": round(threshold, 3), "region_count": len(regions), "regions": regions, "status": "review_required" if regions else "no_large_visual_difference"})

    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
