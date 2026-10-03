"""Candidate extraction layer for the DINO/standard fused difference map."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from dino_feature_diff import fused_evidence


THIN_CORE_RAW_DENSITY_MINIMUM = 0.12


def _candidate_evidence(
    candidate: dict[str, Any], evidence: dict[str, np.ndarray], valid: np.ndarray | None
) -> dict[str, Any]:
    left, top, right, bottom = (int(candidate[key]) for key in ("left", "top", "right", "bottom"))
    usable = np.ones((bottom - top, right - left), dtype=bool) if valid is None else valid[top:bottom, left:right] > 0
    if not np.any(usable):
        return {"pixel_count": 0}

    def statistics(name: str) -> tuple[float, float]:
        values = evidence[name][top:bottom, left:right][usable]
        return float(values.mean()), float(np.percentile(values, 90.0))

    traditional_mean, traditional_p90 = statistics("traditional")
    dino_mean, dino_p90 = statistics("dino")
    agreement_mean, agreement_p90 = statistics("agreement")
    traditional_normalized = evidence["traditional_normalized"][top:bottom, left:right][usable]
    dino_normalized = evidence["dino_normalized"][top:bottom, left:right][usable]
    cross_ratio = float(np.mean((traditional_normalized >= 0.50) & (dino_normalized >= 0.50)))
    return {
        "pixel_count": int(np.count_nonzero(usable)),
        "traditional_mean": round(traditional_mean, 3),
        "traditional_p90": round(traditional_p90, 3),
        "dino_mean": round(dino_mean, 5),
        "dino_p90": round(dino_p90, 5),
        "agreement_mean": round(agreement_mean, 5),
        "agreement_p90": round(agreement_p90, 5),
        "cross_evidence_pixel_ratio": round(cross_ratio, 4),
    }


def candidate_components(
    score: np.ndarray,
    valid: np.ndarray | None = None,
    *,
    preserve_thin_cores: bool = False,
) -> list[dict[str, Any]]:
    """Use high-confidence cores to constrain any low-frequency expansion.

    The broad mask is useful for showing the full extent of a real change, but
    must never create a candidate on its own: AI re-rendering of labels or
    cabinet texture otherwise turns into isolated, misleading rectangles.
    """
    if valid is None:
        usable = np.full(score.shape, 255, dtype=np.uint8)
    else:
        if valid.shape != score.shape:
            raise ValueError(f"valid mask shape {valid.shape} does not match score shape {score.shape}")
        usable = np.uint8(valid > 0) * 255
    values = score[usable > 0]
    if values.size < 100:
        return []
    # Normalize support by available image coverage. Otherwise black warp
    # padding can manufacture a border contour or weaken nearby real evidence.
    support = cv2.GaussianBlur(score * (usable > 0), (51, 51), 0) / np.maximum(
        cv2.GaussianBlur((usable > 0).astype(np.float32), (51, 51), 0), 1e-6
    )
    fine_threshold = max(118.0, float(np.percentile(values, 97.8)))
    broad_threshold = max(78.0, float(np.percentile(support[usable > 0], 92.0)))
    raw_fine = np.uint8((score >= fine_threshold) & (usable > 0)) * 255
    broad = np.uint8((support >= broad_threshold) & (usable > 0)) * 255
    small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    fine = cv2.dilate(cv2.morphologyEx(raw_fine, cv2.MORPH_OPEN, small), small, iterations=2)
    broad = cv2.morphologyEx(broad, cv2.MORPH_CLOSE, large)

    # Keep a broad connected component only when it contains a high-confidence
    # fine core.  This preserves a useful extent around a genuine change while
    # rejecting disconnected blur-only regions.
    component_count, labels = cv2.connectedComponents(broad)
    supported_broad = np.zeros_like(broad)
    thin_core_regions: list[dict[str, Any]] = []
    minimum_core_coverage = 0.40
    minimum = max(220, int(broad.shape[0] * broad.shape[1] * 0.0010))
    for component_id in range(1, component_count):
        component = labels == component_id
        component_area = int(np.count_nonzero(component))
        core_coverage = float(np.count_nonzero(fine[component])) / float(component_area)
        if core_coverage >= minimum_core_coverage:
            supported_broad[component] = 255
        elif preserve_thin_cores and component_area >= minimum:
            raw_density = float(np.count_nonzero(raw_fine[component])) / float(component_area)
            if raw_density >= THIN_CORE_RAW_DENSITY_MINIMUM:
                rows, columns = np.where(component)
                left, top = int(columns.min()), int(rows.min())
                right, bottom = int(columns.max()) + 1, int(rows.max()) + 1
                thin_core_regions.append({
                    "left": left,
                    "top": top,
                    "right": right,
                    "bottom": bottom,
                    "area": round(float(component_area), 1),
                    "difference_score": round(float(score[top:bottom, left:right].mean()), 2),
                    "candidate_kind": "thin_core",
                    "raw_fine_density": round(raw_density, 4),
                })
    mask = cv2.bitwise_or(fine, supported_broad)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions: list[dict[str, Any]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < minimum:
            continue
        left, top, width, height = cv2.boundingRect(contour)
        regions.append({
            "left": left, "top": top, "right": left + width, "bottom": top + height,
            "area": round(area, 1),
            "difference_score": round(float(score[top:top + height, left:left + width].mean()), 2),
        })
    regions.extend(thin_core_regions)
    return sorted(regions, key=lambda item: item["area"] * item["difference_score"], reverse=True)


def fused_components(
    reference: np.ndarray,
    inspection: np.ndarray,
    *,
    require_cross_evidence: bool = False,
    valid: np.ndarray | None = None,
    preserve_thin_cores: bool = False,
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    score, metadata, evidence = fused_evidence(reference, inspection, require_cross_evidence=require_cross_evidence)
    metadata["valid_warp_coverage"] = round(float(np.count_nonzero(valid)) / float(valid.size), 4) if valid is not None else 1.0
    candidates = candidate_components(score, valid, preserve_thin_cores=preserve_thin_cores)
    metadata["thin_core_candidate_count"] = sum(
        candidate.get("candidate_kind") == "thin_core" for candidate in candidates
    )
    for candidate in candidates:
        candidate["evidence_scores"] = _candidate_evidence(candidate, evidence, valid)
    return score, metadata, candidates
