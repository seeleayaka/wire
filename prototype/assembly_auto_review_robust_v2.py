"""V2: automatic local-grid refinement layered over the robust review UI.

The grid is data driven rather than object specific.  Every part of an inspection
area participates.  Stable tiles receive a small Euclidean correction; weak or
large-motion tiles remain visible and are marked uncertain by the parent pipeline.
"""
from __future__ import annotations

import sys
from typing import Any

import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication

import assembly_auto_review_robust as robust


def _ecc_gray(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(cv2.normalize(gray, None, 0.0, 1.0, cv2.NORM_MINMAX).astype(np.float32), (5, 5), 0)


def grid_limited_ecc(reference: np.ndarray, inspection: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    height, width = reference.shape[:2]
    rows, columns = (3, 3) if min(height, width) >= 210 else (2, 2)
    locally_aligned = inspection.copy()
    valid = np.full((height, width), 255, dtype=np.uint8)
    accepted_mask = np.zeros((height, width), dtype=np.uint8)
    tile_reports: list[dict[str, Any]] = []
    accepted_count = 0
    correlations: list[float] = []
    # An 18-pixel context border allows a tile to align on its surrounding fixed
    # structure, while only its centre is copied back.  Thus neighbouring tiles do
    # not create seams and a moved component cannot dictate its own transform.
    margin = 18
    for row in range(rows):
        for column in range(columns):
            x1, x2 = column * width // columns, (column + 1) * width // columns
            y1, y2 = row * height // rows, (row + 1) * height // rows
            ax1, ax2 = max(0, x1 - margin), min(width, x2 + margin)
            ay1, ay2 = max(0, y1 - margin), min(height, y2 + margin)
            warp = np.eye(2, 3, dtype=np.float32)
            report: dict[str, Any] = {"row": row, "column": column, "accepted": False, "correlation": None}
            try:
                correlation, warp = cv2.findTransformECC(
                    _ecc_gray(reference[ay1:ay2, ax1:ax2]),
                    _ecc_gray(inspection[ay1:ay2, ax1:ax2]),
                    warp,
                    cv2.MOTION_EUCLIDEAN,
                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 110, 1e-6),
                    None,
                    5,
                )
                translation = float(np.linalg.norm(warp[:, 2]))
                angle = float(np.degrees(np.arctan2(warp[1, 0], warp[0, 0])))
                accepted = correlation >= 0.76 and translation <= 8.0 and abs(angle) <= 2.5
                report.update(correlation=round(float(correlation), 4), translation_pixels=round(translation, 2), rotation_degrees=round(angle, 2), accepted=bool(accepted))
                correlations.append(float(correlation))
                if accepted:
                    warped = cv2.warpAffine(
                        inspection[ay1:ay2, ax1:ax2], warp, (ax2 - ax1, ay2 - ay1),
                        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE,
                    )
                    locally_aligned[y1:y2, x1:x2] = warped[y1 - ay1:y2 - ay1, x1 - ax1:x2 - ax1]
                    accepted_mask[y1:y2, x1:x2] = 255
                    accepted_count += 1
            except cv2.error:
                report["reason"] = "ecc_not_converged"
            tile_reports.append(report)
    coverage = accepted_count / (rows * columns)
    diagnostic = {
        "method": "local_grid_ecc_euclidean", "grid": [rows, columns], "accepted_tiles": accepted_count,
        "tile_count": rows * columns, "accepted_coverage": round(coverage, 3),
        "correlation": round(float(np.median(correlations)), 4) if correlations else None,
        # Red requires substantially more coverage than yellow.  Yellow does not
        # demand a new photograph; it means the retained orange boxes need review.
        "accepted": bool(coverage >= 0.67 and correlations and np.median(correlations) >= 0.78),
        "tiles": tile_reports,
    }
    # Keep this separate from ``valid``.  ``valid`` describes pixels that are
    # usable for difference scoring; this mask describes pixels whose inspection
    # image was actually replaced by a locally corrected ECC result.
    robust.LAST_LOCAL_ACCEPTED_MASK = accepted_mask
    return locally_aligned, valid, diagnostic


robust._limited_local_ecc = grid_limited_ecc


class GridRobustReview(robust.RobustReview):
    pass


def main() -> None:
    application = QApplication(sys.argv)
    application.setStyle("Fusion")
    window = GridRobustReview()
    window.show()
    raise SystemExit(application.exec_())


if __name__ == "__main__":
    main()
