"""PROTOTYPE ONLY — reference-vs-inspection comparison for a power box.

Question this prototype answers:
Can feature-based image registration followed by a visual-difference map flag
areas worth human review when a correct power-box photo is compared with a new
photo?  It does not decide electrical correctness or diagnose a cable by name.

Run from the project root:
  .\.venv\Scripts\python.exe prototype\power_box_compare_prototype.py ^
    --reference "data\raw\图片1.png" --inspection "data\raw\图片2.png" ^
    --output "output\prototype_compare"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def read_image(path: Path) -> np.ndarray | None:
    """Open Windows paths that contain Chinese characters."""
    try:
        raw = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(raw, cv2.IMREAD_COLOR) if raw.size else None


def write_image(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f"Cannot encode {path}")
    encoded.tofile(str(path))


def align_to_reference(reference: np.ndarray, inspection: np.ndarray) -> tuple[np.ndarray | None, dict[str, float | int | str]]:
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    inspected_gray = cv2.cvtColor(inspection, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=6000, fastThreshold=7)
    ref_keypoints, ref_descriptors = orb.detectAndCompute(ref_gray, None)
    inspected_keypoints, inspected_descriptors = orb.detectAndCompute(inspected_gray, None)
    metrics: dict[str, float | int | str] = {
        "reference_keypoints": len(ref_keypoints),
        "inspection_keypoints": len(inspected_keypoints),
        "good_matches": 0,
        "inliers": 0,
        "inlier_ratio": 0.0,
    }
    if ref_descriptors is None or inspected_descriptors is None:
        metrics["reason"] = "No usable visual features"
        return None, metrics

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(inspected_descriptors, ref_descriptors, k=2)
    good = [first for first, second in pairs if first.distance < 0.72 * second.distance]
    metrics["good_matches"] = len(good)
    if len(good) < 12:
        metrics["reason"] = "Too few matching visual features"
        return None, metrics

    source = np.float32([inspected_keypoints[match.queryIdx].pt for match in good]).reshape(-1, 1, 2)
    destination = np.float32([ref_keypoints[match.trainIdx].pt for match in good]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(source, destination, cv2.RANSAC, 5.0)
    if homography is None or mask is None:
        metrics["reason"] = "Could not estimate image transform"
        return None, metrics
    inliers = int(mask.ravel().sum())
    metrics["inliers"] = inliers
    metrics["inlier_ratio"] = round(inliers / len(good), 4)
    if inliers < 10:
        metrics["reason"] = "Insufficient reliable image alignment"
        return None, metrics

    height, width = reference.shape[:2]
    return cv2.warpPerspective(inspection, homography, (width, height)), metrics


def find_difference_regions(reference: np.ndarray, aligned: np.ndarray) -> tuple[np.ndarray, list[dict[str, int | float]], np.ndarray]:
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    aligned_gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ref_equalized = cv2.GaussianBlur(clahe.apply(ref_gray), (5, 5), 0)
    aligned_equalized = cv2.GaussianBlur(clahe.apply(aligned_gray), (5, 5), 0)
    difference = cv2.absdiff(ref_equalized, aligned_equalized)
    _, mask = cv2.threshold(difference, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(120, int(reference.shape[0] * reference.shape[1] * 0.00008))
    regions: list[dict[str, int | float]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        score = float(difference[y : y + height, x : x + width].mean())
        regions.append({"left": x, "top": y, "right": x + width, "bottom": y + height, "area": round(area, 1), "difference_score": round(score, 2)})
    regions.sort(key=lambda region: float(region["difference_score"]) * float(region["area"]), reverse=True)
    return difference, regions, mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Prototype power-box image comparison.")
    parser.add_argument("--reference", type=Path, required=True, help="Verified-correct reference image")
    parser.add_argument("--inspection", type=Path, required=True, help="New image to compare")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference = read_image(args.reference)
    inspection = read_image(args.inspection)
    if reference is None or inspection is None:
        raise SystemExit("Cannot open the reference or inspection image.")
    args.output.mkdir(parents=True, exist_ok=True)
    aligned, alignment = align_to_reference(reference, inspection)
    report: dict[str, object] = {"prototype": True, "reference": str(args.reference), "inspection": str(args.inspection), "alignment": alignment, "decision": "manual_review", "regions": []}
    if aligned is None:
        report["reason"] = "Images are not aligned reliably enough for comparison. Retake a guided photo."
        (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    difference, regions, mask = find_difference_regions(reference, aligned)
    annotated = aligned.copy()
    for index, region in enumerate(regions, start=1):
        left, top, right, bottom = (int(region[key]) for key in ("left", "top", "right", "bottom"))
        cv2.rectangle(annotated, (left, top), (right, bottom), (0, 0, 255), 3)
        cv2.putText(annotated, f"Review {index}", (left, max(25, top - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    heatmap = cv2.applyColorMap(difference, cv2.COLORMAP_JET)
    report["regions"] = regions
    report["region_count"] = len(regions)
    report["decision"] = "potential_visual_difference" if regions else "no_large_visual_difference"
    report["warning"] = "This is a visual-difference prototype. Every result requires human review."
    write_image(args.output / "aligned.jpg", aligned)
    write_image(args.output / "difference_mask.png", mask)
    write_image(args.output / "difference_heatmap.jpg", heatmap)
    write_image(args.output / "review_regions.jpg", annotated)
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
