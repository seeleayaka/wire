"""V6: use local motion only when its own alignment evidence is sufficient.

When the local grid is unreliable, fall back to the global-homography multiscale
comparison.  This preserves a moved charger's full footprint instead of fragmenting
it into motion artefacts.
"""
from __future__ import annotations

import sys

import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication

import assembly_auto_review_multiscale as multiscale
import assembly_auto_review_robust as robust
import assembly_auto_review_robust_v5 as local_review


def _merge_rois(rois: list[list[float]]) -> list[list[float]]:
    ordered = sorted(rois, key=lambda roi: (roi[2] - roi[0]) * (roi[3] - roi[1]), reverse=True)
    merged: list[list[float]] = []
    for roi in ordered:
        overlaps = False
        for kept in merged:
            left, top = max(roi[0], kept[0]), max(roi[1], kept[1])
            right, bottom = min(roi[2], kept[2]), min(roi[3], kept[3])
            intersection = max(0.0, right - left) * max(0.0, bottom - top)
            smaller = min((roi[2] - roi[0]) * (roi[3] - roi[1]), (kept[2] - kept[0]) * (kept[3] - kept[1]))
            if intersection / max(1e-8, smaller) >= 0.80:
                overlaps = True
                break
        if not overlaps:
            merged.append(roi)
    return merged


def adaptive_regions(reference: np.ndarray, aligned: np.ndarray, rois: list[list[float]]):
    merged = _merge_rois(rois)
    overlay, heat, regions = local_review.merged_motion_regions(reference, aligned, merged)
    diagnostics = list(robust.LAST_DIAGNOSTICS)
    coverage = [item.get("accepted_coverage", 0.0) for item in diagnostics]
    # Motion consistency is useful only with enough reliable local geometry.  A
    # low-coverage grid is evidence to disable that branch, not evidence that every
    # pixel is suspicious.
    if coverage and min(coverage) >= 0.50:
        return overlay, heat, regions
    _, fallback_heat, fallback_regions = multiscale.multiscale_regions(reference, aligned, merged)
    clean = aligned.copy()
    for roi in merged:
        x1, y1, x2, y2 = local_review.motion.perspective.auto.base.pixels(reference, roi)
        cv2.rectangle(clean, (x1, y1), (x2, y2), (0, 215, 255), 2)
    for item in fallback_regions:
        item["confidence"] = "uncertain"
        item["comparison_mode"] = "global_multiscale_fallback"
    fallback_regions.sort(key=lambda item: item["area"] * item["difference_score"], reverse=True)
    for number, item in enumerate(fallback_regions, 1):
        left, top, right, bottom = (int(item[key]) for key in ("left", "top", "right", "bottom"))
        cv2.rectangle(clean, (left, top), (right, bottom), (0, 180, 255), 4)
        cv2.putText(clean, str(number), (left, max(30, top - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 180, 255), 3)
    for item in robust.LAST_DIAGNOSTICS:
        item["comparison_mode"] = "global_multiscale_fallback"
        item["fallback_reason"] = "insufficient_reliable_local_grid"
    return clean, fallback_heat, fallback_regions


robust.robust_regions = adaptive_regions


class AdaptiveReview(local_review.CandidateLocalReview):
    pass


def main() -> None:
    application = QApplication(sys.argv)
    application.setStyle("Fusion")
    window = AdaptiveReview()
    window.show()
    raise SystemExit(application.exec_())


if __name__ == "__main__":
    main()
