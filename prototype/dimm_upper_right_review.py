"""Fixed-view right-upper DIMM visual review prototype.

This tool deliberately produces human-review evidence, not an automatic
pass/fail decision.  Label the right-upper DIMM bank once on a verified-good
reference photo, then inspect fixed-camera photos against that same reference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def read_image(path: Path) -> np.ndarray:
    """Read Windows/Chinese paths safely."""
    raw = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR) if raw.size else None
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f"Cannot encode image: {path}")
    encoded.tofile(str(path))


def align_to(reference: np.ndarray, inspection: np.ndarray) -> tuple[np.ndarray | None, dict[str, Any]]:
    orb = cv2.ORB_create(nfeatures=6000, fastThreshold=7)
    ref_points, ref_desc = orb.detectAndCompute(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), None)
    image_points, image_desc = orb.detectAndCompute(cv2.cvtColor(inspection, cv2.COLOR_BGR2GRAY), None)
    report: dict[str, Any] = {"reference_keypoints": len(ref_points), "inspection_keypoints": len(image_points), "matches": 0, "inliers": 0}
    if ref_desc is None or image_desc is None:
        report["reason"] = "no_visual_descriptors"
        return None, report
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(image_desc, ref_desc, k=2)
    matches = [first for first, second in pairs if first.distance < 0.72 * second.distance]
    report["matches"] = len(matches)
    if len(matches) < 12:
        report["reason"] = "too_few_feature_matches"
        return None, report
    source = np.float32([image_points[item.queryIdx].pt for item in matches]).reshape(-1, 1, 2)
    destination = np.float32([ref_points[item.trainIdx].pt for item in matches]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(source, destination, cv2.RANSAC, 5.0)
    if homography is None or mask is None:
        report["reason"] = "no_geometric_transform"
        return None, report
    report["inliers"] = int(mask.ravel().sum())
    report["inlier_ratio"] = round(report["inliers"] / len(matches), 4)
    if report["inliers"] < 80 or report["inlier_ratio"] < 0.25:
        report["reason"] = "unstable_alignment_retake_photo"
        return None, report
    height, width = reference.shape[:2]
    return cv2.warpPerspective(inspection, homography, (width, height)), report


def normalize_gray(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX), (5, 5), 0)


def roi_pixels(image: np.ndarray, roi: list[float]) -> tuple[int, int, int, int]:
    height, width = image.shape[:2]
    left, top, right, bottom = roi
    return round(left * width), round(top * height), round(right * width), round(bottom * height)


def regions_from_difference(reference: np.ndarray, aligned: np.ndarray, roi: list[float]) -> tuple[np.ndarray, list[dict[str, Any]], float]:
    x1, y1, x2, y2 = roi_pixels(reference, roi)
    ref_crop = normalize_gray(reference[y1:y2, x1:x2])
    inspection_crop = normalize_gray(aligned[y1:y2, x1:x2])
    difference = cv2.absdiff(ref_crop, inspection_crop)
    threshold = max(20.0, float(np.percentile(difference, 99.0)))
    mask = np.uint8(difference >= threshold) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    minimum_area = max(120, int(mask.shape[0] * mask.shape[1] * 0.00035))
    regions: list[dict[str, Any]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < minimum_area:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        score = float(difference[y:y + height, x:x + width].mean())
        regions.append({"left": x + x1, "top": y + y1, "right": x + x1 + width, "bottom": y + y1 + height, "area": round(area, 1), "difference_score": round(score, 2)})
    regions.sort(key=lambda region: float(region["area"]) * float(region["difference_score"]), reverse=True)
    return difference, regions, threshold


def command_label(args: argparse.Namespace) -> None:
    reference = read_image(args.reference)
    print("Drag one rectangle tightly around the RIGHT-UPPER DIMM bank. Enter/Space saves; Esc cancels.")
    boxes = cv2.selectROIs("Label right-upper DIMM bank", reference, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    if len(boxes) != 1:
        raise SystemExit(f"Expected one DIMM-bank rectangle, got {len(boxes)}. Nothing was saved.")
    x, y, width, height = (int(value) for value in boxes[0])
    image_height, image_width = reference.shape[:2]
    payload = {
        "prototype": True,
        "scope": "right_upper_dimm_bank_only",
        "reference_image": str(args.reference.resolve()),
        "reference_size": {"width": image_width, "height": image_height},
        "dimm_bank_roi": [round(x / image_width, 6), round(y / image_height, 6), round((x + width) / image_width, 6), round((y + height) / image_height, 6)],
        "capture_rules": ["same checkpoint", "fixed camera position, direction and distance", "no EXIF rotation", "keep lighting as stable as practical"],
    }
    args.config.parent.mkdir(parents=True, exist_ok=True)
    args.config.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.config}")


def command_inspect(args: argparse.Namespace) -> None:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    reference = read_image(Path(config["reference_image"]))
    inspection = read_image(args.inspection)
    aligned, alignment = align_to(reference, inspection)
    args.output.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"prototype": True, "scope": "right_upper_dimm_bank_only", "reference": config["reference_image"], "inspection": str(args.inspection), "alignment": alignment, "decision": "manual_review"}
    if aligned is None:
        report["reason"] = alignment.get("reason", "alignment_failed")
        (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    difference, regions, threshold = regions_from_difference(reference, aligned, config["dimm_bank_roi"])
    x1, y1, x2, y2 = roi_pixels(reference, config["dimm_bank_roi"])
    overlay = aligned.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 215, 255), 3)
    for index, region in enumerate(regions, start=1):
        left, top, right, bottom = (int(region[key]) for key in ("left", "top", "right", "bottom"))
        cv2.rectangle(overlay, (left, top), (right, bottom), (0, 0, 255), 4)
        cv2.putText(overlay, f"review {index}", (left, max(30, top - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    heatmap = cv2.applyColorMap(difference, cv2.COLORMAP_JET)
    review_crop = overlay[y1:y2, x1:x2]
    reference_crop = reference[y1:y2, x1:x2]
    aligned_crop = aligned[y1:y2, x1:x2]
    panel = np.hstack([reference_crop, aligned_crop, review_crop, heatmap])
    report.update({"difference_threshold": round(threshold, 2), "review_region_count": len(regions), "review_regions": regions, "artifacts": {"aligned": "aligned.jpg", "overview": "overview.jpg", "dimm_review_panel": "dimm_review_panel.jpg"}})
    write_image(args.output / "aligned.jpg", aligned)
    write_image(args.output / "overview.jpg", overlay)
    write_image(args.output / "dimm_review_panel.jpg", panel)
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Right-upper DIMM fixed-view human-review prototype.")
    commands = parser.add_subparsers(dest="command", required=True)
    label = commands.add_parser("label", help="Label the DIMM bank once on a verified-good reference image.")
    label.add_argument("--reference", type=Path, required=True)
    label.add_argument("--config", type=Path, required=True)
    label.set_defaults(handler=command_label)
    inspect = commands.add_parser("inspect", help="Align and highlight visual differences for human review.")
    inspect.add_argument("--config", type=Path, required=True)
    inspect.add_argument("--inspection", type=Path, required=True)
    inspect.add_argument("--output", type=Path, required=True)
    inspect.set_defaults(handler=command_inspect)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
