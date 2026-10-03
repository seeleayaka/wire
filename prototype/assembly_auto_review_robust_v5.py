"""V5: per-candidate local confidence and overlapping-check deduplication."""
from __future__ import annotations

import sys

import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication

import assembly_auto_review_robust as robust
import assembly_auto_review_robust_v4 as motion


_original_candidate_regions = robust._candidate_regions


def candidate_local_confidence(score: np.ndarray, valid: np.ndarray, roi_index: int, origin: tuple[int, int], diagnostic: dict) -> list[dict]:
    """Promote only candidates whose own surrounding tile aligned reliably."""
    candidates = _original_candidate_regions(score, valid, roi_index, origin, diagnostic)
    rows, columns = diagnostic.get("grid", [1, 1])
    tiles = {(item.get("row"), item.get("column")): item for item in diagnostic.get("tiles", [])}
    x0, y0 = origin
    for item in candidates:
        centre_x = (item["left"] + item["right"]) / 2.0 - x0
        centre_y = (item["top"] + item["bottom"]) / 2.0 - y0
        column = min(columns - 1, max(0, int(centre_x * columns / score.shape[1])))
        row = min(rows - 1, max(0, int(centre_y * rows / score.shape[0])))
        tile = tiles.get((row, column), {})
        item["local_tile"] = [row, column]
        item["local_tile_correlation"] = tile.get("correlation")
        # This is intentionally candidate-local.  A bad tile elsewhere in the
        # same large inspection box must not demote a well aligned DIMM candidate.
        if tile.get("accepted") and tile.get("correlation", 0.0) >= 0.80 and item["shape"] == "object_sized" and item["neighbourhood_score"] >= 18.0:
            item["confidence"] = "high"
    return candidates


def _overlap_ratio(first: list[float], second: list[float]) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    smaller = min((first[2] - first[0]) * (first[3] - first[1]), (second[2] - second[0]) * (second[3] - second[1]))
    return intersection / max(1e-8, smaller)


def merged_motion_regions(reference: np.ndarray, aligned: np.ndarray, rois: list[list[float]]):
    # A nested setup rectangle commonly appears when users adjust a check area.
    # Scan it once, using the larger rectangle, rather than duplicate candidates.
    ordered = sorted(rois, key=lambda roi: (roi[2] - roi[0]) * (roi[3] - roi[1]), reverse=True)
    merged: list[list[float]] = []
    for roi in ordered:
        if not any(_overlap_ratio(roi, kept) >= 0.80 for kept in merged):
            merged.append(roi)
    overlay, heat, regions = motion.motion_aware_regions(reference, aligned, merged)
    # Rebuild the overlay so only the merged check rectangles and final candidate
    # boxes are visible; the parent function drew before the final ordering.
    overlay = aligned.copy()
    for roi in merged:
        x1, y1, x2, y2 = motion.perspective.auto.base.pixels(reference, roi)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 215, 255), 2)
    for number, item in enumerate(regions, 1):
        left, top, right, bottom = (int(item[key]) for key in ("left", "top", "right", "bottom"))
        colour = (0, 0, 255) if item["confidence"] == "high" else (0, 180, 255)
        cv2.rectangle(overlay, (left, top), (right, bottom), colour, 4)
        cv2.putText(overlay, str(number), (left, max(30, top - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 3)
    return overlay, heat, regions


robust._candidate_regions = candidate_local_confidence
robust.robust_regions = merged_motion_regions


class CandidateLocalReview(motion.MotionAwareReview):
    pass


def main() -> None:
    application = QApplication(sys.argv)
    application.setStyle("Fusion")
    window = CandidateLocalReview()
    window.show()
    raise SystemExit(application.exec_())


if __name__ == "__main__":
    main()
