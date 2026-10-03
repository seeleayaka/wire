"""Independent two-band alignment experiment for a deep control cabinet.

The complete cabinet is not treated as one plane.  After the existing global
SIFT/MAGSAC homography, this probe estimates one extra homography for each
predefined device band.  The results are shown as separate crops only; they are
never stitched into a synthetic full-cabinet image.  A straight line remains a
straight line inside every band because every correction is a single 3x3
homography, not a mesh, optical flow, or liquid warp.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "prototype"))
import assembly_auto_review_robust_v3 as perspective  # noqa: E402


# Cabinet-5-specific visible device planes.  They deliberately stop at the
# large horizontal ducts, so a cross-band wire is never made continuous by an
# artificial stitched warp.
BANDS: tuple[tuple[str, tuple[float, float, float, float]], ...] = (
    ("upper_device_plane", (0.07, 0.09, 0.94, 0.37)),
    ("lower_device_plane", (0.07, 0.40, 0.94, 0.70)),
)


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    okay, encoded = cv2.imencode(path.suffix or ".png", image)
    if not okay:
        raise ValueError(f"Cannot encode image: {path}")
    encoded.tofile(str(path))


def bounds_from_normalized(image: np.ndarray, roi: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    height, width = image.shape[:2]
    left, top, right, bottom = roi
    x1, x2 = round(left * width), round(right * width)
    y1, y2 = round(top * height), round(bottom * height)
    if x2 - x1 < 40 or y2 - y1 < 40:
        raise ValueError(f"Band ROI too small: {roi}")
    return x1, y1, x2, y2


def affine_like_geometry_ok(transform: np.ndarray, width: int, height: int) -> tuple[bool, dict[str, float]]:
    """Reject local H transforms that are implausibly large for a post-global correction."""
    corners = np.float32([[[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]])
    mapped = cv2.perspectiveTransform(corners, transform).reshape(-1, 2)
    original = corners.reshape(-1, 2)
    area = abs(cv2.contourArea(mapped.astype(np.float32))) / max(1.0, float(width * height))
    max_shift = float(np.linalg.norm(mapped - original, axis=1).max())
    centroid_offset = float(np.linalg.norm(mapped.mean(axis=0) - original.mean(axis=0)))
    # Raw homography coefficients are expressed in image pixels.  Their
    # condition number therefore changes with crop resolution and is not a
    # geometry gate.  Measure the equivalent mapping in centred unit
    # coordinates, as the global alignment code does.
    normalize = np.array([[1.0 / width, 0.0, -0.5], [0.0, 1.0 / height, -0.5], [0.0, 0.0, 1.0]])
    normalized = normalize @ transform @ np.linalg.inv(normalize)
    normalized /= normalized[2, 2] if abs(normalized[2, 2]) > 1e-9 else 1.0
    condition = float(np.linalg.cond(normalized))
    values = {
        "local_warped_area_ratio": round(area, 4),
        "local_max_corner_shift_pixels": round(max_shift, 3),
        "local_centroid_offset_pixels": round(centroid_offset, 3),
        "local_normalized_transform_condition": round(condition, 3),
    }
    good = 0.72 <= area <= 1.32 and max_shift <= max(26.0, 0.16 * max(width, height)) and centroid_offset <= max(18.0, 0.10 * max(width, height)) and condition <= 45.0
    return good, values


def estimate_local_homography(reference_crop: np.ndarray, base_crop: np.ndarray) -> tuple[np.ndarray | None, dict[str, Any]]:
    sift = cv2.SIFT_create(nfeatures=3500, contrastThreshold=0.014, edgeThreshold=12)
    ref_points, ref_descriptors = sift.detectAndCompute(cv2.cvtColor(reference_crop, cv2.COLOR_BGR2GRAY), None)
    test_points, test_descriptors = sift.detectAndCompute(cv2.cvtColor(base_crop, cv2.COLOR_BGR2GRAY), None)
    report: dict[str, Any] = {
        "method": "local_band_sift_usac_magsac_homography_after_global_H",
        "reference_keypoints": len(ref_points),
        "base_aligned_keypoints": len(test_points),
        "matches": 0,
        "inliers": 0,
        "applied": False,
    }
    if ref_descriptors is None or test_descriptors is None:
        report["reason"] = "insufficient_local_texture"
        return None, report
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(test_descriptors, ref_descriptors, k=2)
    good = [first for first, second in pairs if first.distance < 0.70 * second.distance]
    report["matches"] = len(good)
    if len(good) < 28:
        report["reason"] = "too_few_local_matches"
        return None, report
    source = np.float32([test_points[item.queryIdx].pt for item in good]).reshape(-1, 1, 2)
    destination = np.float32([ref_points[item.trainIdx].pt for item in good]).reshape(-1, 1, 2)
    method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
    transform, mask = cv2.findHomography(source, destination, method=method, ransacReprojThreshold=2.5, maxIters=10000, confidence=0.995)
    if transform is None or mask is None:
        report["reason"] = "no_stable_local_homography"
        return None, report
    inliers = mask.ravel().astype(bool)
    report["inliers"] = int(inliers.sum())
    report["inlier_ratio"] = round(float(inliers.mean()), 4)
    projected = cv2.perspectiveTransform(source, transform)
    errors = np.linalg.norm(projected.reshape(-1, 2) - destination.reshape(-1, 2), axis=1)
    report["median_reprojection_error_pixels"] = round(float(np.median(errors[inliers])), 3) if report["inliers"] else None
    geometry_ok, geometry = affine_like_geometry_ok(transform, reference_crop.shape[1], reference_crop.shape[0])
    report.update(geometry)
    reliable = report["inliers"] >= 22 and report["inlier_ratio"] >= 0.30 and report["median_reprojection_error_pixels"] is not None and report["median_reprojection_error_pixels"] <= 1.8 and geometry_ok
    if not reliable:
        report["reason"] = "local_homography_rejected_by_match_or_geometry_gate"
        return None, report
    report.update(applied=True, reason="passed")
    return transform, report


def local_to_full_transform(local: np.ndarray, bounds: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, _, _ = bounds
    to_local = np.array([[1.0, 0.0, -x1], [0.0, 1.0, -y1], [0.0, 0.0, 1.0]])
    from_local = np.array([[1.0, 0.0, x1], [0.0, 1.0, y1], [0.0, 0.0, 1.0]])
    full = from_local @ local @ to_local
    return full / full[2, 2]


def edge_overlap_f1(reference: np.ndarray, aligned: np.ndarray) -> float | None:
    ref_gray = cv2.GaussianBlur(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    aligned_gray = cv2.GaussianBlur(cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    ref_edges = cv2.Canny(ref_gray, 45, 120)
    test_edges = cv2.Canny(aligned_gray, 45, 120)
    border = 5
    ref_edges[:border, :] = ref_edges[-border:, :] = 0
    ref_edges[:, :border] = ref_edges[:, -border:] = 0
    test_edges[:border, :] = test_edges[-border:, :] = 0
    test_edges[:, :border] = test_edges[:, -border:] = 0
    ref_count, test_count = int(np.count_nonzero(ref_edges)), int(np.count_nonzero(test_edges))
    if not ref_count or not test_count:
        return None
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    ref_near = cv2.dilate(ref_edges, kernel)
    test_near = cv2.dilate(test_edges, kernel)
    precision = float(np.count_nonzero((test_edges > 0) & (ref_near > 0)) / test_count)
    recall = float(np.count_nonzero((ref_edges > 0) & (test_near > 0)) / ref_count)
    return round(float(2 * precision * recall / (precision + recall)), 4) if precision + recall else 0.0


def labelled_panel(image: np.ndarray, label: str) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (min(result.shape[1] - 1, 470), 34), (25, 25, 25), thickness=-1)
    cv2.putText(result, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return result


def run(reference_path: Path, inspection_path: Path, output: Path) -> dict[str, Any]:
    reference = read_image(reference_path)
    inspection = read_image(inspection_path)
    base, base_report = perspective.automatic_homography(reference, inspection)
    output.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "kind": "separate_local_band_homography_probe",
        "geometry_contract": "global H, then separate local 3x3 H inside each displayed band; no mesh, optical flow, liquid warp, or stitched full image",
        "reference": str(reference_path),
        "inspection": str(inspection_path),
        "bands": [],
        "base_alignment": base_report,
    }
    if base is None:
        result["decision"] = "global_alignment_uncertain_manual_review"
        (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    for name, normalized_roi in BANDS:
        bounds = bounds_from_normalized(reference, normalized_roi)
        x1, y1, x2, y2 = bounds
        reference_crop = reference[y1:y2, x1:x2]
        base_crop = base[y1:y2, x1:x2]
        local, local_report = estimate_local_homography(reference_crop, base_crop)
        band: dict[str, Any] = {"name": name, "normalized_roi": normalized_roi, "pixel_bounds": bounds, "local_alignment": local_report}
        if local is None:
            local_crop = base_crop
            band["decision"] = "retain_global_alignment_for_this_band"
        else:
            full_transform = local_to_full_transform(local, bounds)
            local_full = cv2.warpPerspective(base, full_transform, (reference.shape[1], reference.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            local_crop = local_full[y1:y2, x1:x2]
            band["decision"] = "local_band_alignment_for_manual_view"
        band["base_edge_overlap_f1"] = edge_overlap_f1(reference_crop, base_crop)
        band["local_edge_overlap_f1"] = edge_overlap_f1(reference_crop, local_crop)
        result["bands"].append(band)
        panels = [
            labelled_panel(reference_crop, f"{name}: reference"),
            labelled_panel(cv2.addWeighted(reference_crop, 0.5, base_crop, 0.5, 0), f"{name}: global H blend"),
            labelled_panel(cv2.addWeighted(reference_crop, 0.5, local_crop, 0.5, 0), f"{name}: local H blend"),
        ]
        write_image(output / f"{name}_comparison.jpg", np.hstack(panels))
        write_image(output / f"{name}_reference.jpg", reference_crop)
        write_image(output / f"{name}_global_aligned.jpg", base_crop)
        write_image(output / f"{name}_local_aligned.jpg", local_crop)

    result["decision"] = "separate_local_band_alignment_manual_review"
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--inspection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args.reference, args.inspection, args.output)
    print(json.dumps({"decision": result["decision"], "bands": result["bands"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
