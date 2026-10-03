"""Template review with geometric validation before any perspective warp."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

import assembly_template_review_app as base
from dimm_review_app_fixed import RoiCanvas


if not hasattr(RoiCanvas, "normalized_roi"):
    RoiCanvas.normalized_roi = RoiCanvas.roi  # type: ignore[attr-defined]


def geometry_is_valid(
    homography: np.ndarray, reference: np.ndarray, inspection: np.ndarray,
    template_bounds: tuple[int, int, int, int], report: dict[str, Any],
) -> bool:
    x1, y1, x2, y2 = template_bounds
    try:
        inverse = np.linalg.inv(homography)
        condition = float(np.linalg.cond(homography))
        corners = np.float32([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]])
        mapped = cv2.perspectiveTransform(corners, inverse).reshape(-1, 2)
    except (cv2.error, np.linalg.LinAlgError):
        report["reason"] = "singular_template_transform"
        return False
    height, width = inspection.shape[:2]
    template_area = float((x2 - x1) * (y2 - y1))
    mapped_area = abs(float(cv2.contourArea(mapped.astype(np.float32))))
    area_ratio = mapped_area / max(template_area, 1.0)
    report["homography_condition"] = round(condition, 2)
    report["projected_template_area_ratio"] = round(area_ratio, 3)
    report["projected_template_corners"] = np.round(mapped, 1).tolist()
    inside = bool(np.all(mapped[:, 0] >= -0.15 * width) and np.all(mapped[:, 0] <= 1.15 * width)
                  and np.all(mapped[:, 1] >= -0.15 * height) and np.all(mapped[:, 1] <= 1.15 * height))
    convex = bool(cv2.isContourConvex(mapped.astype(np.float32)))
    if not inside or not convex or not (0.25 <= area_ratio <= 3.5) or condition > 1e10:
        report["reason"] = "geometrically_implausible_transform_reselect_template"
        return False
    return True


def locate_with_geometry(
    reference: np.ndarray, inspection: np.ndarray, template_roi: list[float]
) -> tuple[np.ndarray | None, dict[str, Any]]:
    x1, y1, x2, y2 = base.pixels(reference, template_roi)
    template = reference[y1:y2, x1:x2]
    sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.015, edgeThreshold=12)
    template_kp, template_desc = sift.detectAndCompute(cv2.cvtColor(template, cv2.COLOR_BGR2GRAY), None)
    image_kp, image_desc = sift.detectAndCompute(cv2.cvtColor(inspection, cv2.COLOR_BGR2GRAY), None)
    report: dict[str, Any] = {
        "method": "template_sift_magsac_geometric_validation",
        "template_keypoints": len(template_kp), "inspection_keypoints": len(image_kp),
        "matches": 0, "inliers": 0,
    }
    if template_desc is None or image_desc is None:
        report["reason"] = "template_has_too_little_texture"
        return None, report
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(image_desc, template_desc, k=2)
    good = [first for first, second in pairs if first.distance < 0.72 * second.distance]
    report["matches"] = len(good)
    if len(good) < 18:
        report["reason"] = "too_few_template_matches"
        return None, report
    source = np.float32([image_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    destination = np.float32([[template_kp[m.trainIdx].pt[0] + x1, template_kp[m.trainIdx].pt[1] + y1] for m in good]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(source, destination, getattr(cv2, "USAC_MAGSAC", cv2.RANSAC), 5.0)
    if homography is None or mask is None:
        report["reason"] = "no_template_transform"
        return None, report
    inlier_mask = mask.ravel().astype(bool)
    report["inliers"] = int(inlier_mask.sum())
    report["inlier_ratio"] = round(report["inliers"] / len(good), 4)
    if not report["inliers"]:
        report["reason"] = "no_template_inliers"
        return None, report
    projected = cv2.perspectiveTransform(source, homography)
    error = np.linalg.norm(projected.reshape(-1, 2) - destination.reshape(-1, 2), axis=1)
    report["median_reprojection_error"] = round(float(np.median(error[inlier_mask])), 2)
    if report["inliers"] < 18 or report["median_reprojection_error"] > 3.5:
        report["reason"] = "unstable_template_location_retake_photo"
        return None, report
    if not geometry_is_valid(homography, reference, inspection, (x1, y1, x2, y2), report):
        return None, report
    height, width = reference.shape[:2]
    return cv2.warpPerspective(inspection, homography, (width, height)), report


base.locate = locate_with_geometry


if __name__ == "__main__":
    base.main()
