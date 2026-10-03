"""Enhanced template-review entry point: ORB first, SIFT/MAGSAC fallback.

Mobile photos of a repeating power strip can make ORB reject an otherwise
valid local template.  The original fast path is preserved, while SIFT is
used only when that path cannot pass its safety checks.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

import assembly_template_review_app as base
from dimm_review_app_fixed import RoiCanvas


if not hasattr(RoiCanvas, "normalized_roi"):
    RoiCanvas.normalized_roi = RoiCanvas.roi  # type: ignore[attr-defined]


def sift_fallback(
    reference: np.ndarray, inspection: np.ndarray, template_roi: list[float]
) -> tuple[np.ndarray | None, dict[str, Any]]:
    x1, y1, x2, y2 = base.pixels(reference, template_roi)
    template = reference[y1:y2, x1:x2]
    sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.015, edgeThreshold=12)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    gray_inspection = cv2.cvtColor(inspection, cv2.COLOR_BGR2GRAY)
    template_kp, template_desc = sift.detectAndCompute(gray_template, None)
    image_kp, image_desc = sift.detectAndCompute(gray_inspection, None)
    report: dict[str, Any] = {
        "method": "template_sift_magsac_fallback",
        "template_keypoints": len(template_kp),
        "inspection_keypoints": len(image_kp),
        "matches": 0,
        "inliers": 0,
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
    destination = np.float32(
        [[template_kp[m.trainIdx].pt[0] + x1, template_kp[m.trainIdx].pt[1] + y1] for m in good]
    ).reshape(-1, 1, 2)
    method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
    homography, mask = cv2.findHomography(source, destination, method, 5.0)
    if homography is None or mask is None:
        report["reason"] = "no_template_transform"
        return None, report
    inlier_mask = mask.ravel().astype(bool)
    report["inliers"] = int(inlier_mask.sum())
    report["inlier_ratio"] = round(report["inliers"] / len(good), 4)
    if report["inliers"]:
        projected = cv2.perspectiveTransform(source, homography)
        error = np.linalg.norm(projected.reshape(-1, 2) - destination.reshape(-1, 2), axis=1)
        report["median_reprojection_error"] = round(float(np.median(error[inlier_mask])), 2)
    else:
        report["median_reprojection_error"] = None
    if report["inliers"] < 18 or report["median_reprojection_error"] is None or report["median_reprojection_error"] > 3.5:
        report["reason"] = "unstable_template_location_retake_photo"
        return None, report
    height, width = reference.shape[:2]
    return cv2.warpPerspective(inspection, homography, (width, height)), report


def robust_locate(
    reference: np.ndarray, inspection: np.ndarray, template_roi: list[float]
) -> tuple[np.ndarray | None, dict[str, Any]]:
    aligned, report = base.locate(reference, inspection, template_roi)
    if aligned is not None:
        return aligned, report
    fallback_aligned, fallback_report = sift_fallback(reference, inspection, template_roi)
    fallback_report["orb_first_attempt"] = report
    return fallback_aligned, fallback_report


base.locate = robust_locate


if __name__ == "__main__":
    base.main()
