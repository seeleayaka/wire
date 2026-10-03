"""Automatic review with both fine detail and broad low-contrast change masks."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

import assembly_auto_review as app


def multiscale_regions(reference: np.ndarray, aligned: np.ndarray, rois: list[list[float]]) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    overlay = aligned.copy()
    heat = np.zeros_like(reference)
    regions: list[dict[str, Any]] = []
    for roi_index, roi in enumerate(rois, 1):
        x1, y1, x2, y2 = app.base.pixels(reference, roi)
        ref_part, test_part = reference[y1:y2, x1:x2], aligned[y1:y2, x1:x2]
        def gray(image: np.ndarray) -> np.ndarray:
            return cv2.GaussianBlur(cv2.normalize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None, 0, 255, cv2.NORM_MINMAX), (3, 3), 0)
        ref_gray, test_gray = gray(ref_part), gray(test_part)
        tone = cv2.absdiff(ref_gray, test_gray).astype(np.float32)
        edge = cv2.absdiff(cv2.Canny(ref_gray, 45, 120), cv2.Canny(test_gray, 45, 120)).astype(np.float32)
        ref_lab = cv2.GaussianBlur(cv2.cvtColor(ref_part, cv2.COLOR_BGR2LAB), (3, 3), 0)
        test_lab = cv2.GaussianBlur(cv2.cvtColor(test_part, cv2.COLOR_BGR2LAB), (3, 3), 0)
        chroma = cv2.absdiff(ref_lab[:, :, 1:], test_lab[:, :, 1:]).mean(axis=2).astype(np.float32)
        score = np.maximum.reduce((tone, edge * 0.80, chroma * 1.35)).astype(np.uint8)
        # Fine mask catches high-contrast edges; broad mask preserves a large
        # displaced object even when its per-pixel contrast is moderate.
        fine = np.uint8(score >= max(20.0, float(np.percentile(score, 97.8)))) * 255
        broad_score = cv2.GaussianBlur(score, (51, 51), 0)
        broad = np.uint8(broad_score >= max(18.0, float(np.percentile(broad_score, 92.0)))) * 255
        small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        large_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        fine = cv2.dilate(cv2.morphologyEx(fine, cv2.MORPH_OPEN, small_kernel), small_kernel, iterations=2)
        broad = cv2.morphologyEx(broad, cv2.MORPH_CLOSE, large_kernel)
        mask = cv2.bitwise_or(fine, broad)
        heat[y1:y2, x1:x2] = cv2.applyColorMap(np.maximum(score, broad_score), cv2.COLORMAP_JET)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 215, 255), 2)
        minimum = max(220, int(mask.shape[0] * mask.shape[1] * 0.0010))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < minimum:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            regions.append({"check_region": roi_index, "left": x + x1, "top": y + y1, "right": x + x1 + width, "bottom": y + y1 + height, "area": round(area, 1), "difference_score": round(float(score[y:y + height, x:x + width].mean()), 2)})
    regions.sort(key=lambda item: item["area"] * item["difference_score"], reverse=True)
    for number, item in enumerate(regions, 1):
        left, top, right, bottom = (int(item[key]) for key in ("left", "top", "right", "bottom"))
        cv2.rectangle(overlay, (left, top), (right, bottom), (0, 0, 255), 4)
        cv2.putText(overlay, str(number), (left, max(30, top - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
    return overlay, heat, regions


app.colour_aware_structural_regions = multiscale_regions


if __name__ == "__main__":
    app.main()
