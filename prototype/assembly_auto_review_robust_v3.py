"""V3: perspective-aware coarse registration for the local-grid robust review."""
from __future__ import annotations

from typing import Any
import sys

import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication

import assembly_auto_review_robust_v2 as grid

auto = grid.robust.auto
# Set by every successful global homography. Consumers that compare the warped
# image must ignore pixels not covered by the source image.
auto.LAST_WARP_VALID_MASK: np.ndarray | None = None


def automatic_homography(reference: np.ndarray, inspection: np.ndarray) -> tuple[np.ndarray | None, dict[str, Any]]:
    auto.LAST_WARP_VALID_MASK = None
    sift = cv2.SIFT_create(nfeatures=9000, contrastThreshold=0.014, edgeThreshold=12)
    ref_keypoints, ref_descriptors = sift.detectAndCompute(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), None)
    test_keypoints, test_descriptors = sift.detectAndCompute(cv2.cvtColor(inspection, cv2.COLOR_BGR2GRAY), None)
    report: dict[str, Any] = {
        "method": "automatic_full_frame_sift_homography",
        "reference_keypoints": len(ref_keypoints), "inspection_keypoints": len(test_keypoints),
        "matches": 0, "inliers": 0,
    }
    if ref_descriptors is None or test_descriptors is None:
        report["reason"] = "too_little_visual_texture"
        report["alignment_quality"] = {"reliable": False, "reason": report["reason"]}
        return None, report
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(test_descriptors, ref_descriptors, k=2)
    good = [first for first, second in pairs if first.distance < 0.70 * second.distance]
    report["matches"] = len(good)
    if len(good) < 60:
        report["reason"] = "too_few_full_frame_matches"
        report["alignment_quality"] = {"reliable": False, "reason": report["reason"]}
        return None, report
    source = np.float32([test_keypoints[match.queryIdx].pt for match in good]).reshape(-1, 1, 2)
    destination = np.float32([ref_keypoints[match.trainIdx].pt for match in good]).reshape(-1, 1, 2)
    method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
    transform, mask = cv2.findHomography(source, destination, method=method, ransacReprojThreshold=4.0, maxIters=10000, confidence=0.995)
    if transform is None or mask is None:
        report["reason"] = "no_stable_perspective_transform"
        report["alignment_quality"] = {"reliable": False, "reason": report["reason"]}
        return None, report
    inlier_mask = mask.ravel().astype(bool)
    report["inliers"] = int(inlier_mask.sum())
    report["inlier_ratio"] = round(report["inliers"] / len(good), 4)
    projected = cv2.perspectiveTransform(source, transform)
    errors = np.linalg.norm(projected.reshape(-1, 2) - destination.reshape(-1, 2), axis=1)
    report["median_reprojection_error"] = round(float(np.median(errors[inlier_mask])), 2) if report["inliers"] else None
    height, width = reference.shape[:2]
    cells = {(min(3, int(destination[index, 0, 0] * 4 / width)), min(3, int(destination[index, 0, 1] * 4 / height))) for index in np.where(inlier_mask)[0]}
    report["inlier_grid_cells"] = len(cells)
    corners = np.float32([[[0, 0], [inspection.shape[1] - 1, 0], [inspection.shape[1] - 1, inspection.shape[0] - 1], [0, inspection.shape[0] - 1]]])
    warped_corners = cv2.perspectiveTransform(corners, transform).reshape(-1, 2)
    warped_area = abs(cv2.contourArea(warped_corners.astype(np.float32)))
    # This measures how much of the *reference frame* the aligned inspection
    # occupies.  Do not divide by the original inspection area: a perfectly
    # valid AI-edited or resized image can have different pixel dimensions.
    reference_area_ratio = warped_area / float(max(1, reference.shape[0] * reference.shape[1]))
    source_scale_area_ratio = warped_area / float(max(1, inspection.shape[0] * inspection.shape[1]))
    report["warped_area_ratio"] = round(reference_area_ratio, 4)
    report["source_to_reference_area_scale"] = round(source_scale_area_ratio, 4)
    finite_transform = bool(np.isfinite(transform).all() and np.isfinite(warped_corners).all())
    # Raw homography coefficients are expressed in image pixels, so their
    # condition number changes with resolution and is not a reliable gate.
    # Measure it only after mapping both images into centred unit coordinates.
    source_normalize = np.array([[1.0 / inspection.shape[1], 0.0, -0.5], [0.0, 1.0 / inspection.shape[0], -0.5], [0.0, 0.0, 1.0]])
    destination_normalize = np.array([[1.0 / width, 0.0, -0.5], [0.0, 1.0 / height, -0.5], [0.0, 0.0, 1.0]])
    normalized_transform = destination_normalize @ transform @ np.linalg.inv(source_normalize) if finite_transform else transform
    normalized_transform /= normalized_transform[2, 2] if finite_transform and abs(normalized_transform[2, 2]) > 1e-9 else 1.0
    transform_condition = float(np.linalg.cond(normalized_transform)) if finite_transform else float("inf")
    side_lengths = np.linalg.norm(warped_corners - np.roll(warped_corners, -1, axis=0), axis=1)
    shortest_side = float(side_lengths.min()) if finite_transform else 0.0
    reference_diagonal = float(np.hypot(width, height))
    centroid = warped_corners.mean(axis=0) if finite_transform else np.array([np.inf, np.inf])
    centroid_offset = float(np.linalg.norm(centroid - np.array([width / 2.0, height / 2.0])) / max(1.0, reference_diagonal))
    checks = {
        "finite_transform": finite_transform,
        "normalized_condition_number": round(transform_condition, 2) if np.isfinite(transform_condition) else None,
        "corner_shortest_side_pixels": round(shortest_side, 2),
        "corner_centroid_offset_ratio": round(centroid_offset, 4) if np.isfinite(centroid_offset) else None,
        "inlier_count_ok": report["inliers"] >= 70,
        "inlier_ratio_ok": report["inlier_ratio"] >= 0.22,
        "spatial_coverage_ok": len(cells) >= 4,
        "reprojection_error_ok": report["median_reprojection_error"] is not None and report["median_reprojection_error"] <= 2.5,
        "area_ratio_ok": 0.60 <= reference_area_ratio <= 1.50,
        "geometry_ok": finite_transform and transform_condition <= 50.0 and shortest_side >= min(width, height) * 0.15 and centroid_offset <= 0.75,
    }
    reliable = all(value for key, value in checks.items() if key.endswith("_ok"))
    report["alignment_quality"] = {
        "reliable": reliable,
        "checks": checks,
        "units": {"normalized_condition_number": "dimensionless_in_unit_coordinates", "corner_shortest_side_pixels": "pixels", "corner_centroid_offset_ratio": "reference_diagonal_ratio", "warped_area_ratio": "warped_area_over_reference_area", "source_to_reference_area_scale": "warped_area_over_original_inspection_area"},
    }
    if not reliable:
        report["reason"] = "automatic_perspective_alignment_uncertain_manual_review"
        report["alignment_quality"]["reason"] = report["reason"]
        return None, report
    # Constant warp padding is not evidence of an assembly difference.
    valid = cv2.warpPerspective(
        np.full(inspection.shape[:2], 255, np.uint8), transform, (width, height),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    valid = cv2.erode(valid, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    auto.LAST_WARP_VALID_MASK = valid
    report["valid_warp_coverage"] = round(float(np.count_nonzero(valid)) / float(valid.size), 4)
    report["alignment_quality"]["reason"] = "passed"
    report["source_to_reference_homography"] = transform.tolist()
    return cv2.warpPerspective(inspection, transform, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT), report


auto.automatic_affine = automatic_homography


class PerspectiveGridReview(grid.GridRobustReview):
    pass


def main() -> None:
    application = QApplication(sys.argv)
    application.setStyle("Fusion")
    window = PerspectiveGridReview()
    window.show()
    raise SystemExit(application.exec_())


if __name__ == "__main__":
    main()
