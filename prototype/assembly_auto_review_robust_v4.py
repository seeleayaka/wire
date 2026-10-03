"""V4: add dense motion-consistency evidence to the perspective robust review."""
from __future__ import annotations

import sys
from typing import Any

import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication

import assembly_auto_review_robust as robust
import assembly_auto_review_robust_v3 as perspective


def motion_aware_regions(reference: np.ndarray, aligned: np.ndarray, rois: list[list[float]]) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Score appearance change and motion that is inconsistent with its neighbours."""
    robust.LAST_DIAGNOSTICS = []
    local_aligned_full = aligned.copy()
    local_accepted_full = np.zeros(aligned.shape[:2], dtype=np.uint8)
    local_alignment_parts: list[dict[str, Any]] = []
    overlay, heat, regions = aligned.copy(), np.zeros_like(reference), []
    for roi_index, roi in enumerate(rois, 1):
        x1, y1, x2, y2 = perspective.auto.base.pixels(reference, roi)
        ref_part, test_part = reference[y1:y2, x1:x2], aligned[y1:y2, x1:x2]
        local, valid, diagnostic = robust._limited_local_ecc(ref_part, test_part)
        local_accepted = getattr(robust, "LAST_LOCAL_ACCEPTED_MASK", np.zeros(valid.shape, dtype=np.uint8))
        if local_accepted.shape == local.shape[:2]:
            local_aligned_full[y1:y2, x1:x2] = local
            local_accepted_full[y1:y2, x1:x2] = local_accepted
            local_alignment_parts.append({
                "bounds": [x1, y1, x2, y2],
                "aligned": local,
                "accepted_mask": local_accepted.copy(),
            })
        else:
            local_alignment_parts.append({"bounds": [x1, y1, x2, y2]})
        diagnostic.update(check_region=roi_index, bounds=[x1, y1, x2, y2])
        ref_gray, local_gray = robust._illumination_normalized_gray(ref_part), robust._illumination_normalized_gray(local)
        structure = robust._ssim_difference(ref_gray, local_gray)
        gradient = np.clip(np.abs(robust._gradient(ref_gray) - robust._gradient(local_gray)) * 255.0, 0.0, 255.0)
        ref_lab = cv2.cvtColor(ref_part, cv2.COLOR_BGR2LAB).astype(np.float32)
        local_lab = cv2.cvtColor(local, cv2.COLOR_BGR2LAB).astype(np.float32)
        colour = np.clip(np.linalg.norm(ref_lab - local_lab, axis=2) * 1.20, 0.0, 255.0)
        appearance = np.clip(structure * 0.55 + colour * 0.25 + gradient * 0.20, 0.0, 255.0)
        # Camera/viewpoint movement is smooth over neighbouring pixels.  A component
        # moved relative to its mounting has a residual flow different from that
        # smooth field.  This is generic: it does not know whether the component is
        # a DIMM, heatsink, cable, connector, or something else.
        flow = cv2.calcOpticalFlowFarneback(
            np.uint8(ref_gray * 255), np.uint8(local_gray * 255), None,
            0.5, 4, 35, 5, 7, 1.5, 0,
        )
        smooth_flow = cv2.GaussianBlur(flow, (0, 0), 35.0)
        motion = np.linalg.norm(flow - smooth_flow, axis=2)
        usable_motion = motion[valid > 0]
        motion_limit = float(np.percentile(usable_motion, 96.0)) if usable_motion.size else 1.0
        motion_score = np.clip(motion * 255.0 / max(1.0, motion_limit), 0.0, 255.0)
        score = np.clip(appearance * 0.70 + motion_score * 0.30, 0.0, 255.0).astype(np.float32)
        score[valid == 0] = 0.0
        diagnostic["motion_residual_p96"] = round(motion_limit, 3)
        robust.LAST_DIAGNOSTICS.append(diagnostic)
        heat[y1:y2, x1:x2] = cv2.applyColorMap(np.uint8(score), cv2.COLORMAP_JET)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 215, 255), 2)
        candidates = robust._candidate_regions(score, valid, roi_index, (x1, y1), diagnostic)
        for candidate in candidates:
            left, top, right, bottom = (candidate[key] for key in ("left", "top", "right", "bottom"))
            local_motion = motion_score[top - y1:bottom - y1, left - x1:right - x1]
            candidate["motion_inconsistency"] = round(float(local_motion.mean()), 2)
        regions.extend(candidates)
    # The DINO fusion layer consumes these diagnostics after the motion pass.
    # Expose the image and mask separately so an unreliable local grid can fall
    # back to the global homography without losing the valid local corrections.
    robust.LAST_LOCAL_ALIGNED = local_aligned_full
    robust.LAST_LOCAL_ACCEPTED_MASK = local_accepted_full
    robust.LAST_LOCAL_ALIGNMENT_PARTS = local_alignment_parts
    # Object-sized candidates with a movement residual rise above repetitive
    # fine-edge residuals, even when both remain yellow due to low alignment trust.
    regions.sort(key=lambda item: (item["confidence"] == "high", item.get("motion_inconsistency", 0.0), item["area"] * item["neighbourhood_score"]), reverse=True)
    for number, item in enumerate(regions, 1):
        left, top, right, bottom = (int(item[key]) for key in ("left", "top", "right", "bottom"))
        colour = (0, 0, 255) if item["confidence"] == "high" else (0, 180, 255)
        cv2.rectangle(overlay, (left, top), (right, bottom), colour, 4)
        cv2.putText(overlay, str(number), (left, max(30, top - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 3)
    return overlay, heat, regions


robust.robust_regions = motion_aware_regions


class MotionAwareReview(perspective.PerspectiveGridReview):
    pass


def main() -> None:
    application = QApplication(sys.argv)
    application.setStyle("Fusion")
    window = MotionAwareReview()
    window.show()
    raise SystemExit(application.exec_())


if __name__ == "__main__":
    main()
