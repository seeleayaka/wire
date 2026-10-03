"""Overlapping-tile DINO review for a large, user-defined inspection region.

The UI continues to expose one check region.  This module keeps that meaning,
but runs a whole-region coarse pass plus smaller overlapping passes internally
so a local defect is not normalized away by a large amount of cabinet texture.
"""
from __future__ import annotations

from math import ceil, sqrt
from typing import Any

import cv2
import numpy as np

from dino_feature_diff_v2 import fused_components


DEFAULT_WIDTH_RATIO = 0.22
DEFAULT_HEIGHT_RATIO = 0.26
MIN_TILE_EDGE = 224
MAX_TILE_EDGE = 420
DEFAULT_OVERLAP_RATIO = 0.30
DEFAULT_MAX_TILES = 12
REFINEMENT_WIDTH_RATIO = 0.38
REFINEMENT_HEIGHT_RATIO = 0.36
REFINEMENT_MAX_TILES = 9
# A cabinet-wide ROI is intentionally sensitive.  Keep the UI concise while
# retaining every suppressed observation in report metadata for later review.
LARGE_ROI_MAX_CANDIDATES = 3
DOMINANT_CONSENSUS_MIN_SOURCES = 8
DOMINANT_CONSENSUS_MIN_SCALES = 3
DOMINANT_CONSENSUS_MIN_CROSS_RATIO = 0.90
TILE_TRIGGER_ROI_AREA_RATIO = 0.08
REPETITIVE_RAW_DENSITY_MINIMUM = 0.15
REPETITIVE_ASPECT_RATIO = 1.8
REPETITIVE_MIN_SOURCE_TILES = 2


def _axis_starts(length: int, tile_length: int, overlap_ratio: float) -> list[int]:
    """Produce starts that always include the final edge-aligned tile."""
    if tile_length >= length:
        return [0]
    step = max(1, round(tile_length * (1.0 - overlap_ratio)))
    last = length - tile_length
    starts = list(range(0, last + 1, step))
    if starts[-1] != last:
        starts.append(last)
    return starts


def _windows(width: int, height: int, tile_width: int, tile_height: int, overlap_ratio: float) -> list[tuple[int, int, int, int]]:
    return [
        (left, top, left + tile_width, top + tile_height)
        for top in _axis_starts(height, tile_height, overlap_ratio)
        for left in _axis_starts(width, tile_width, overlap_ratio)
    ]


def tile_plan(
    roi_width: int,
    roi_height: int,
    *,
    reference_size: tuple[int, int] | None = None,
    tile_width: int | None = None,
    tile_height: int | None = None,
    overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
    max_tiles: int = DEFAULT_MAX_TILES,
) -> tuple[tuple[int, int], list[tuple[int, int, int, int]]]:
    """Return bounded overlapping windows, enlarging tiles if there are too many."""
    if roi_width <= 0 or roi_height <= 0:
        raise ValueError("ROI dimensions must be positive")
    if not 0.0 <= overlap_ratio < 1.0:
        raise ValueError("overlap_ratio must be in [0, 1)")
    if max_tiles < 1:
        raise ValueError("max_tiles must be positive")
    full_width, full_height = reference_size or (roi_width, roi_height)
    target_width = tile_width or round(full_width * DEFAULT_WIDTH_RATIO)
    target_height = tile_height or round(full_height * DEFAULT_HEIGHT_RATIO)
    target_width = min(roi_width, max(MIN_TILE_EDGE, min(MAX_TILE_EDGE, target_width)))
    target_height = min(roi_height, max(MIN_TILE_EDGE, min(MAX_TILE_EDGE, target_height)))
    windows = _windows(roi_width, roi_height, target_width, target_height, overlap_ratio)
    while len(windows) > max_tiles and (target_width < roi_width or target_height < roi_height):
        scale = max(1.08, sqrt(len(windows) / float(max_tiles)))
        grown_width = min(roi_width, max(target_width + 1, ceil(target_width * scale)))
        grown_height = min(roi_height, max(target_height + 1, ceil(target_height * scale)))
        if (grown_width, grown_height) == (target_width, target_height):
            break
        target_width, target_height = grown_width, grown_height
        windows = _windows(roi_width, roi_height, target_width, target_height, overlap_ratio)
    return (target_width, target_height), windows


def refinement_plan(
    roi_width: int,
    roi_height: int,
    *,
    reference_size: tuple[int, int],
) -> tuple[tuple[int, int], list[tuple[int, int, int, int]]]:
    """Use a wider second scale so small changes crossing first-pass tiles persist."""
    full_width, full_height = reference_size
    tile_width = min(roi_width, max(MIN_TILE_EDGE, round(full_width * REFINEMENT_WIDTH_RATIO)))
    tile_height = min(roi_height, max(MIN_TILE_EDGE, round(full_height * REFINEMENT_HEIGHT_RATIO)))

    def evenly_spaced_starts(length: int, tile_length: int) -> list[int]:
        if tile_length >= length:
            return [0]
        last = length - tile_length
        return sorted({0, round(last / 2.0), last})

    windows = [
        (left, top, left + tile_width, top + tile_height)
        for top in evenly_spaced_starts(roi_height, tile_height)
        for left in evenly_spaced_starts(roi_width, tile_width)
    ]
    return (tile_width, tile_height), windows


def _iou(first: dict[str, Any], second: dict[str, Any]) -> float:
    left, top = max(int(first["left"]), int(second["left"])), max(int(first["top"]), int(second["top"]))
    right, bottom = min(int(first["right"]), int(second["right"])), min(int(first["bottom"]), int(second["bottom"]))
    intersection = max(0, right - left) * max(0, bottom - top)
    area_first = max(1, int(first["right"]) - int(first["left"])) * max(1, int(first["bottom"]) - int(first["top"]))
    area_second = max(1, int(second["right"]) - int(second["left"])) * max(1, int(second["bottom"]) - int(second["top"]))
    return intersection / float(area_first + area_second - intersection)


def _near_touching(first: dict[str, Any], second: dict[str, Any]) -> bool:
    first_width = max(1, int(first["right"]) - int(first["left"]))
    first_height = max(1, int(first["bottom"]) - int(first["top"]))
    second_width = max(1, int(second["right"]) - int(second["left"]))
    second_height = max(1, int(second["bottom"]) - int(second["top"]))
    overlap_x = max(0, min(int(first["right"]), int(second["right"])) - max(int(first["left"]), int(second["left"])))
    overlap_y = max(0, min(int(first["bottom"]), int(second["bottom"])) - max(int(first["top"]), int(second["top"])))
    gap_x = max(0, max(int(first["left"]), int(second["left"])) - min(int(first["right"]), int(second["right"])))
    gap_y = max(0, max(int(first["top"]), int(second["top"])) - min(int(first["bottom"]), int(second["bottom"])))
    if overlap_x >= 0.50 * min(first_width, second_width) and gap_y <= 0.15 * min(first_height, second_height):
        return True
    if overlap_y >= 0.50 * min(first_height, second_height) and gap_x <= 0.15 * min(first_width, second_width):
        return True
    if overlap_x == 0 and overlap_y == 0:
        return False
    first_center = ((int(first["left"]) + int(first["right"])) / 2.0, (int(first["top"]) + int(first["bottom"])) / 2.0)
    second_center = ((int(second["left"]) + int(second["right"])) / 2.0, (int(second["top"]) + int(second["bottom"])) / 2.0)
    distance = float(np.hypot(first_center[0] - second_center[0], first_center[1] - second_center[1]))
    smallest_edge = min(
        first_width,
        first_height,
        second_width,
        second_height,
    )
    return distance <= max(1.0, 0.15 * smallest_edge)


def _contains(first: dict[str, Any], second: dict[str, Any]) -> bool:
    """Treat an almost-complete nested candidate as the same finding."""
    left, top = max(int(first["left"]), int(second["left"])), max(int(first["top"]), int(second["top"]))
    right, bottom = min(int(first["right"]), int(second["right"])), min(int(first["bottom"]), int(second["bottom"]))
    intersection = max(0, right - left) * max(0, bottom - top)
    second_area = max(1, int(second["right"]) - int(second["left"])) * max(1, int(second["bottom"]) - int(second["top"]))
    return intersection / float(second_area) >= 0.90


def _merge_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge overlapping findings from the whole pass and adjacent tiles."""
    pending = [dict(candidate) for candidate in candidates]
    groups: list[list[dict[str, Any]]] = []
    while pending:
        group = [pending.pop(0)]
        changed = True
        while changed:
            changed = False
            kept: list[dict[str, Any]] = []
            for candidate in pending:
                if any(
                    _iou(candidate, member) >= 0.25
                    or _near_touching(candidate, member)
                    or _contains(candidate, member)
                    or _contains(member, candidate)
                    for member in group
                ):
                    group.append(candidate)
                    changed = True
                else:
                    kept.append(candidate)
            pending = kept
        groups.append(group)
    merged: list[dict[str, Any]] = []
    for group in groups:
        source_tiles = sorted({tile for member in group for tile in member["source_tiles"]})
        tile_members = [member for member in group if member.get("source") in {"tile", "refinement"}]
        evidence_members = [member["evidence_scores"] for member in group if member.get("evidence_scores")]
        evidence_scores = {}
        if evidence_members:
            for key in (
                "traditional_mean", "traditional_p90", "dino_mean", "dino_p90",
                "agreement_mean", "agreement_p90", "cross_evidence_pixel_ratio",
            ):
                evidence_scores[f"{key}_max"] = round(max(float(item.get(key, 0.0)) for item in evidence_members), 5)
            evidence_scores["observation_count"] = len(evidence_members)
        merged.append(
            {
                "left": min(int(member["left"]) for member in group),
                "top": min(int(member["top"]) for member in group),
                "right": max(int(member["right"]) for member in group),
                "bottom": max(int(member["bottom"]) for member in group),
                "area": round(sum(float(member["area"]) for member in group), 1),
                "difference_score": round(max(float(member["difference_score"]) for member in group), 2),
                "source_tiles": source_tiles,
                "evidence_scores": evidence_scores,
                "evidence_summary": {
                    "source_tile_count": len([tile for tile in source_tiles if tile != "whole_roi"]),
                    "primary_tile_count": len([tile for tile in source_tiles if tile.startswith("tile_")]),
                    "refinement_tile_count": len([tile for tile in source_tiles if tile.startswith("refine_")]),
                    "evidence_scale_count": sum((
                        "whole_roi" in source_tiles,
                        any(tile.startswith("tile_") for tile in source_tiles),
                        any(tile.startswith("refine_") for tile in source_tiles),
                    )),
                    "whole_roi_overlap": "whole_roi" in source_tiles,
                    "tile_edge_hits": sum(bool(member.get("touches_tile_edge")) for member in tile_members),
                    "tile_edge_only": bool(tile_members) and all(bool(member.get("touches_tile_edge")) for member in tile_members),
                    "merged_observation_count": len(group),
                    "thin_core_observation_count": sum(
                        member.get("candidate_kind") == "thin_core" for member in group
                    ),
                },
            }
        )
    # Edge-only, single-tile candidates remain visible but are naturally placed
    # after findings corroborated by another tile or the coarse whole-ROI pass.
    return sorted(
        merged,
        key=lambda item: (
            item["evidence_summary"]["whole_roi_overlap"] or item["evidence_summary"]["source_tile_count"] > 1,
            not item["evidence_summary"]["tile_edge_only"],
            item["area"] * item["difference_score"],
        ),
        reverse=True,
    )


def _annotate_roi_edges(candidates: list[dict[str, Any]], roi_width: int, roi_height: int) -> None:
    """Record whether a merged box reaches the physical review boundary."""
    for candidate in candidates:
        candidate["evidence_summary"]["touches_roi_edge"] = bool(
            int(candidate["left"]) <= 0
            or int(candidate["top"]) <= 0
            or int(candidate["right"]) >= roi_width
            or int(candidate["bottom"]) >= roi_height
        )


def _touches_tile_edge(candidate: dict[str, Any], tile_width: int, tile_height: int) -> bool:
    margin = 2
    return (
        int(candidate["left"]) <= margin
        or int(candidate["top"]) <= margin
        or int(candidate["right"]) >= tile_width - margin
        or int(candidate["bottom"]) >= tile_height - margin
    )


def _repetitive_tile_candidates(score: np.ndarray, valid: np.ndarray | None) -> list[dict[str, Any]]:
    """Keep dense, discontinuous fine evidence that normal blob cleanup removes.

    The normal candidate path intentionally opens its fine mask before merging it
    with broad support.  That is appropriate for isolated label/rail noise, but
    it erases parallel thin evidence from repeated structures.  This secondary
    path retains the raw fine mask and requires its density inside an independent
    broad-support component instead.
    """
    if valid is None:
        usable = np.full(score.shape, 255, dtype=np.uint8)
    else:
        if valid.shape != score.shape:
            raise ValueError("valid mask shape must match the score shape")
        usable = np.uint8(valid > 0) * 255
    values = score[usable > 0]
    if values.size < 100:
        return []
    coverage = usable > 0
    support = cv2.GaussianBlur(score * coverage, (51, 51), 0) / np.maximum(
        cv2.GaussianBlur(coverage.astype(np.float32), (51, 51), 0), 1e-6
    )
    raw_threshold = max(118.0, float(np.percentile(values, 97.8)))
    broad_threshold = max(78.0, float(np.percentile(support[coverage], 92.0)))
    raw_fine = np.uint8((score >= raw_threshold) & coverage) * 255
    broad = np.uint8((support >= broad_threshold) & coverage) * 255
    broad = cv2.morphologyEx(
        broad,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
    )
    minimum_area = max(220, int(score.shape[0] * score.shape[1] * 0.0010))
    component_count, labels = cv2.connectedComponents(broad)
    candidates: list[dict[str, Any]] = []
    for component_id in range(1, component_count):
        component = labels == component_id
        component_area = int(np.count_nonzero(component))
        if component_area < minimum_area:
            continue
        raw_density = float(np.count_nonzero(raw_fine[component])) / float(component_area)
        if raw_density < REPETITIVE_RAW_DENSITY_MINIMUM:
            continue
        rows, columns = np.where(component)
        left, top = int(columns.min()), int(rows.min())
        right, bottom = int(columns.max()) + 1, int(rows.max()) + 1
        candidates.append(
            {
                "left": left,
                "top": top,
                "right": right,
                "bottom": bottom,
                "area": round(float(component_area), 1),
                "difference_score": round(float(score[top:bottom, left:right].mean()), 2),
                "raw_fine_density": round(raw_density, 4),
            }
        )
    return candidates


def _repetitive_seed(candidate: dict[str, Any], roi_width: int, roi_height: int) -> dict[str, Any] | None:
    """Expand an elongated local finding to its likely repeated-structure band."""
    left, top, right, bottom = (int(candidate[key]) for key in ("left", "top", "right", "bottom"))
    width, height = right - left, bottom - top
    if min(width, height) <= 0:
        return None
    details = {key: value for key, value in candidate.items() if key not in {"left", "top", "right", "bottom"}}
    if height / float(width) >= REPETITIVE_ASPECT_RATIO:
        return {
            **details,
            "left": max(0, left - width),
            "top": max(0, round(top - height * 0.12)),
            "right": min(roi_width, right + width),
            "bottom": min(roi_height, round(bottom + height * 0.12)),
            "repeat_orientation": "vertical",
        }
    if width / float(height) >= REPETITIVE_ASPECT_RATIO:
        return {
            **details,
            "left": max(0, round(left - width * 0.12)),
            "top": max(0, top - height),
            "right": min(roi_width, round(right + width * 0.12)),
            "bottom": min(roi_height, bottom + height),
            "repeat_orientation": "horizontal",
        }
    return None


def _repeat_compatible(first: dict[str, Any], second: dict[str, Any]) -> bool:
    """Link nearby elongated bands only when their short axes agree."""
    if first["repeat_orientation"] != second["repeat_orientation"]:
        return False
    if first["repeat_orientation"] == "vertical":
        cross_overlap = max(0, min(first["right"], second["right"]) - max(first["left"], second["left"]))
        short_axis = min(first["right"] - first["left"], second["right"] - second["left"])
        long_gap = max(0, max(first["top"], second["top"]) - min(first["bottom"], second["bottom"]))
    else:
        cross_overlap = max(0, min(first["bottom"], second["bottom"]) - max(first["top"], second["top"]))
        short_axis = min(first["bottom"] - first["top"], second["bottom"] - second["top"])
        long_gap = max(0, max(first["left"], second["left"]) - min(first["right"], second["right"]))
    return cross_overlap >= 0.35 * short_axis and long_gap <= 1.5 * short_axis


def _near_repeat_group(candidate: dict[str, Any], group: list[dict[str, Any]], orientation: str) -> bool:
    """Absorb small raw fragments that complete a corroborated repeated band."""
    left = min(int(item["left"]) for item in group)
    top = min(int(item["top"]) for item in group)
    right = max(int(item["right"]) for item in group)
    bottom = max(int(item["bottom"]) for item in group)
    center_x = (int(candidate["left"]) + int(candidate["right"])) / 2.0
    center_y = (int(candidate["top"]) + int(candidate["bottom"])) / 2.0
    if orientation == "vertical":
        short_axis = right - left
        long_gap = max(0, int(candidate["top"]) - bottom, top - int(candidate["bottom"]))
        return left <= center_x <= right and long_gap <= 1.5 * short_axis
    short_axis = bottom - top
    long_gap = max(0, int(candidate["left"]) - right, left - int(candidate["right"]))
    return top <= center_y <= bottom and long_gap <= 1.5 * short_axis


def _group_repetitive_candidates(
    candidates: list[dict[str, Any]], roi_width: int, roi_height: int
) -> list[dict[str, Any]]:
    """Build evidence-backed, group-level rectangles for repeated structures."""
    seeds = [seed for candidate in candidates if (seed := _repetitive_seed(candidate, roi_width, roi_height)) is not None]
    pending = list(seeds)
    groups: list[list[dict[str, Any]]] = []
    while pending:
        group = [pending.pop(0)]
        changed = True
        while changed:
            changed = False
            kept: list[dict[str, Any]] = []
            for candidate in pending:
                if any(_repeat_compatible(candidate, member) for member in group):
                    group.append(candidate)
                    changed = True
                else:
                    kept.append(candidate)
            pending = kept
        groups.append(group)

    merged: list[dict[str, Any]] = []
    for group in groups:
        source_tiles = sorted({tile for item in group for tile in item["source_tiles"]})
        if len(source_tiles) < REPETITIVE_MIN_SOURCE_TILES:
            continue
        orientation = str(group[0]["repeat_orientation"])
        members = list(group)
        for candidate in candidates:
            if candidate not in members and _near_repeat_group(candidate, members, orientation):
                members.append(candidate)
        left = min(int(item["left"]) for item in members)
        top = min(int(item["top"]) for item in members)
        right = max(int(item["right"]) for item in members)
        bottom = max(int(item["bottom"]) for item in members)
        source_tiles = sorted({tile for item in members for tile in item["source_tiles"]})
        tile_members = [item for item in members if item.get("source") == "repetitive_tile"]
        merged.append(
            {
                "left": left,
                "top": top,
                "right": right,
                "bottom": bottom,
                "area": round(float((right - left) * (bottom - top)), 1),
                "difference_score": round(max(float(item["difference_score"]) for item in members), 2),
                "source_tiles": source_tiles,
                "evidence_summary": {
                    "source_tile_count": len(source_tiles),
                    "whole_roi_overlap": False,
                    "tile_edge_hits": sum(bool(item.get("touches_tile_edge")) for item in tile_members),
                    "tile_edge_only": bool(tile_members) and all(bool(item.get("touches_tile_edge")) for item in tile_members),
                    "merged_observation_count": len(members),
                    "evidence_kind": "repetitive_structure",
                    "repeat_orientation": orientation,
                    "raw_fine_density_max": round(max(float(item["raw_fine_density"]) for item in members), 4),
                },
            }
        )
    return merged


def _publish_candidates(
    ordinary: list[dict[str, Any]], repetitive: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Prefer corroborated group evidence over weak edge fragments in one ROI."""
    if not repetitive:
        return ordinary, []
    published: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    for candidate in ordinary:
        evidence = candidate["evidence_summary"]
        weak_edge_fragment = (
            evidence["source_tile_count"] == 1
            and not evidence["whole_roi_overlap"]
            and evidence["tile_edge_only"]
        )
        (suppressed if weak_edge_fragment else published).append(candidate)
    published.extend(repetitive)
    published.sort(
        key=lambda item: (
            item["evidence_summary"]["whole_roi_overlap"] or item["evidence_summary"]["source_tile_count"] > 1,
            not item["evidence_summary"]["tile_edge_only"],
            item["area"] * item["difference_score"],
        ),
        reverse=True,
    )
    return published, suppressed


def _display_candidates_for_large_roi(candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep a cabinet-scale ROI from exposing every sensitive tile artefact.

    Whole-ROI cross evidence remains authoritative for a large cabinet.  A
    tile-only finding can still be shown when the overlapping windows provide
    strong independent spatial corroboration.  Less-supported findings remain
    in metadata for diagnostics instead of disappearing silently.
    """
    displayed: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    for candidate in candidates:
        evidence = candidate["evidence_summary"]
        refinement_count = int(evidence.get("refinement_tile_count", 0))
        primary_count = int(evidence.get("primary_tile_count", evidence.get("source_tile_count", 0)))
        cross_scale_refinement = refinement_count >= 1 and (evidence["whole_roi_overlap"] or primary_count >= 1)
        repeated_refinement = refinement_count >= 2
        cross_ratio = float(candidate.get("evidence_scores", {}).get("cross_evidence_pixel_ratio_max", 0.0))
        strong_cross_evidence = cross_ratio >= 0.50
        thin_core_count = int(evidence.get("thin_core_observation_count", 0))
        thin_core_only = (
            thin_core_count > 0
            and thin_core_count == int(evidence.get("merged_observation_count", 0))
        )
        # A real inspection item can sit on an operator-drawn ROI boundary.
        # Whole-ROI support is sufficient evidence in that case; suppress only
        # tile-only findings that lack broad overlapping corroboration.
        ordinary_support = (
            evidence["whole_roi_overlap"]
            or primary_count >= 4
            or (strong_cross_evidence and (cross_scale_refinement or repeated_refinement))
        )
        publish = (strong_cross_evidence and cross_scale_refinement) if thin_core_only else ordinary_support
        if publish:
            displayed.append(candidate)
        else:
            suppressed.append(candidate)
    return displayed, suppressed


def _large_roi_candidate_budget(candidates: list[dict[str, Any]]) -> tuple[int, str]:
    """Choose a conservative display budget from automatic evidence only."""
    if not candidates:
        return 0, "no_candidates"
    evidence = candidates[0]["evidence_summary"]
    cross_ratio = float(candidates[0].get("evidence_scores", {}).get("cross_evidence_pixel_ratio_max", 0.0))
    dominant_consensus = (
        int(evidence.get("evidence_scale_count", 1)) >= DOMINANT_CONSENSUS_MIN_SCALES
        and int(evidence.get("source_tile_count", 0)) >= DOMINANT_CONSENSUS_MIN_SOURCES
        and cross_ratio >= DOMINANT_CONSENSUS_MIN_CROSS_RATIO
    )
    if dominant_consensus:
        return 1, "dominant_multiscale_consensus"
    return min(LARGE_ROI_MAX_CANDIDATES, len(candidates)), "operator_noise_budget"


def _limit_candidates_for_large_roi(
    candidates: list[dict[str, Any]], budget: int | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep the strongest cross-scale findings within an evidence-driven budget."""
    def strong_thin_core(candidate: dict[str, Any]) -> bool:
        evidence = candidate["evidence_summary"]
        observation_count = int(evidence.get("merged_observation_count", 0))
        thin_core_count = int(evidence.get("thin_core_observation_count", 0))
        cross_ratio = float(candidate.get("evidence_scores", {}).get("cross_evidence_pixel_ratio_max", 0.0))
        return observation_count > 0 and thin_core_count == observation_count and cross_ratio >= 0.50

    ordered = sorted(
        candidates,
        key=lambda item: (
            int(item["evidence_summary"].get("evidence_scale_count", 1)),
            strong_thin_core(item),
            int(item["evidence_summary"].get("source_tile_count", 0)),
            float(item.get("evidence_scores", {}).get("cross_evidence_pixel_ratio_max", 0.0)),
            float(item["area"]) * float(item["difference_score"]),
        ),
        reverse=True,
    )
    if budget is None:
        budget, _reason = _large_roi_candidate_budget(ordered)
    return ordered[:budget], ordered[budget:]


def tiled_fused_components(
    reference_roi: np.ndarray,
    aligned_roi: np.ndarray,
    valid_roi: np.ndarray | None,
    *,
    reference_size: tuple[int, int] | None = None,
    tile_width: int | None = None,
    tile_height: int | None = None,
    overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
    enable_whole_roi: bool = True,
    require_whole_cross_evidence: bool = False,
    max_tiles: int = DEFAULT_MAX_TILES,
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    """Fuse whole-ROI and overlapping-tile review in ROI-relative coordinates."""
    if reference_roi.shape != aligned_roi.shape:
        raise ValueError("reference and aligned ROI shapes must match")
    if valid_roi is not None and valid_roi.shape != reference_roi.shape[:2]:
        raise ValueError("valid ROI shape must match the image ROI")
    roi_height, roi_width = reference_roi.shape[:2]
    (actual_width, actual_height), windows = tile_plan(
        roi_width,
        roi_height,
        reference_size=reference_size,
        tile_width=tile_width,
        tile_height=tile_height,
        overlap_ratio=overlap_ratio,
        max_tiles=max_tiles,
    )
    # One window identical to the ROI adds no spatial-scale benefit and would
    # duplicate the coarse pass.
    enable_tiles = len(windows) > 1
    combined_score = np.zeros((roi_height, roi_width), dtype=np.float32)
    raw_candidates: list[dict[str, Any]] = []
    repetitive_tile_candidates: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {
        "mode": "whole_roi_plus_overlapping_tiles" if enable_tiles and enable_whole_roi else ("overlapping_tiles_only" if enable_tiles else "whole_roi_only"),
        "roi_size_pixels": [roi_width, roi_height],
        "tile_size_pixels": [actual_width, actual_height],
        "overlap_ratio": overlap_ratio,
        "tile_count": len(windows) if enable_tiles else 0,
        "whole_roi_candidate_count": 0,
        "tile_candidate_count": 0,
        "repetitive_tile_candidate_count": 0,
        "repetitive_group_candidate_count": 0,
        "repetitive_grouping_enabled": not require_whole_cross_evidence,
        "refinement_enabled": require_whole_cross_evidence,
        "refinement_tile_count": 0,
        "merged_candidate_count": 0,
        "valid_warp_coverage": round(float(np.count_nonzero(valid_roi)) / float(valid_roi.size), 4) if valid_roi is not None else 1.0,
    }
    if enable_whole_roi:
        whole_score, whole_metadata, whole_candidates = fused_components(
            reference_roi,
            aligned_roi,
            require_cross_evidence=require_whole_cross_evidence,
            valid=valid_roi,
        )
        combined_score = np.maximum(combined_score, whole_score)
        metadata["whole_roi"] = whole_metadata
        metadata["whole_roi_candidate_count"] = len(whole_candidates)
        for candidate in whole_candidates:
            raw_candidates.append({**candidate, "source": "whole_roi", "source_tiles": ["whole_roi"], "touches_tile_edge": False})
    tile_reports: list[dict[str, Any]] = []
    if enable_tiles:
        for index, (left, top, right, bottom) in enumerate(windows, 1):
            tile_valid = valid_roi[top:bottom, left:right] if valid_roi is not None else None
            tile_score, tile_metadata, tile_candidates = fused_components(
                reference_roi[top:bottom, left:right],
                aligned_roi[top:bottom, left:right],
                require_cross_evidence=False,
                valid=tile_valid,
                preserve_thin_cores=require_whole_cross_evidence,
            )
            combined_score[top:bottom, left:right] = np.maximum(combined_score[top:bottom, left:right], tile_score)
            tile_id = f"tile_{index:02d}"
            # Repeated thin structures describe compact hardware banks well,
            # but whole-cabinet images contain rails and cable runs that create
            # the same pattern without representing one local inspection item.
            tile_repetitive_candidates = (
                _repetitive_tile_candidates(tile_score, tile_valid)
                if not require_whole_cross_evidence
                else []
            )
            tile_reports.append({
                "id": tile_id,
                "bounds": [left, top, right, bottom],
                "candidate_count": len(tile_candidates),
                "repetitive_candidate_count": len(tile_repetitive_candidates),
                "dino": tile_metadata,
            })
            for candidate in tile_candidates:
                raw_candidates.append(
                    {
                        **candidate,
                        "left": int(candidate["left"]) + left,
                        "top": int(candidate["top"]) + top,
                        "right": int(candidate["right"]) + left,
                        "bottom": int(candidate["bottom"]) + top,
                        "source": "tile",
                        "source_tiles": [tile_id],
                        "touches_tile_edge": _touches_tile_edge(candidate, right - left, bottom - top),
                    }
                )
            for candidate in tile_repetitive_candidates:
                repetitive_tile_candidates.append(
                    {
                        **candidate,
                        "left": int(candidate["left"]) + left,
                        "top": int(candidate["top"]) + top,
                        "right": int(candidate["right"]) + left,
                        "bottom": int(candidate["bottom"]) + top,
                        "source": "repetitive_tile",
                        "source_tiles": [tile_id],
                        "touches_tile_edge": _touches_tile_edge(candidate, right - left, bottom - top),
                    }
                )
        metadata["tiles"] = tile_reports
        metadata["tile_candidate_count"] = sum(item["candidate_count"] for item in tile_reports)
        metadata["repetitive_tile_candidate_count"] = sum(item["repetitive_candidate_count"] for item in tile_reports)
    if require_whole_cross_evidence:
        (_refinement_width, _refinement_height), refinement_windows = refinement_plan(
            roi_width,
            roi_height,
            reference_size=reference_size or (roi_width, roi_height),
        )
        refinement_reports: list[dict[str, Any]] = []
        for index, (left, top, right, bottom) in enumerate(refinement_windows, 1):
            refinement_valid = valid_roi[top:bottom, left:right] if valid_roi is not None else None
            refinement_score, refinement_metadata, refinement_candidates = fused_components(
                reference_roi[top:bottom, left:right],
                aligned_roi[top:bottom, left:right],
                require_cross_evidence=False,
                valid=refinement_valid,
                preserve_thin_cores=True,
            )
            combined_score[top:bottom, left:right] = np.maximum(
                combined_score[top:bottom, left:right], refinement_score
            )
            refinement_id = f"refine_{index:02d}"
            refinement_reports.append({
                "id": refinement_id,
                "bounds": [left, top, right, bottom],
                "candidate_count": len(refinement_candidates),
                "dino": refinement_metadata,
            })
            for candidate in refinement_candidates:
                raw_candidates.append({
                    **candidate,
                    "left": int(candidate["left"]) + left,
                    "top": int(candidate["top"]) + top,
                    "right": int(candidate["right"]) + left,
                    "bottom": int(candidate["bottom"]) + top,
                    "source": "refinement",
                    "source_tiles": [refinement_id],
                    "touches_tile_edge": _touches_tile_edge(candidate, right - left, bottom - top),
                })
        metadata["refinement_tiles"] = refinement_reports
        metadata["refinement_tile_count"] = len(refinement_reports)
        metadata["refinement_candidate_count"] = sum(item["candidate_count"] for item in refinement_reports)
    if valid_roi is not None:
        combined_score = combined_score.copy()
        combined_score[valid_roi <= 0] = 0.0
    ordinary_merged = _merge_candidates(raw_candidates)
    repetitive_merged = _group_repetitive_candidates(repetitive_tile_candidates, roi_width, roi_height)
    _annotate_roi_edges(ordinary_merged, roi_width, roi_height)
    _annotate_roi_edges(repetitive_merged, roi_width, roi_height)
    merged, weak_edge_suppressed = _publish_candidates(ordinary_merged, repetitive_merged)
    metadata["repetitive_group_candidate_count"] = len(repetitive_merged)
    metadata["suppressed_weak_edge_candidate_count"] = len(weak_edge_suppressed)
    if weak_edge_suppressed:
        metadata["suppressed_weak_edge_candidates"] = [
            {
                "left": candidate["left"], "top": candidate["top"],
                "right": candidate["right"], "bottom": candidate["bottom"],
                "source_tiles": candidate["source_tiles"],
                "evidence_summary": candidate["evidence_summary"],
                "evidence_scores": candidate.get("evidence_scores", {}),
            }
            for candidate in weak_edge_suppressed
        ]
    # Only cabinet-scale regions use this stricter publication policy.  A
    # medium-sized ROI (for example memory2's 16.2% target) keeps its sensitive
    # single-tile findings, which is the point of introducing tiling.
    if require_whole_cross_evidence:
        displayed, suppressed = _display_candidates_for_large_roi(merged)
        metadata["unfiltered_merged_candidate_count"] = len(merged)
        metadata["suppressed_candidate_count"] = len(suppressed)
        metadata["suppressed_candidates"] = [
            {
                "left": candidate["left"], "top": candidate["top"],
                "right": candidate["right"], "bottom": candidate["bottom"],
                "source_tiles": candidate["source_tiles"],
                "evidence_summary": candidate["evidence_summary"],
                "evidence_scores": candidate.get("evidence_scores", {}),
            }
            for candidate in suppressed
        ]
        merged = displayed
        candidate_budget, candidate_budget_reason = _large_roi_candidate_budget(merged)
        merged, budget_suppressed = _limit_candidates_for_large_roi(merged, budget=candidate_budget)
        metadata["candidate_budget"] = candidate_budget
        metadata["candidate_budget_reason"] = candidate_budget_reason
        metadata["budget_suppressed_candidate_count"] = len(budget_suppressed)
        metadata["budget_suppressed_candidates"] = [
            {
                "left": candidate["left"], "top": candidate["top"],
                "right": candidate["right"], "bottom": candidate["bottom"],
                "source_tiles": candidate["source_tiles"],
                "evidence_summary": candidate["evidence_summary"],
                "evidence_scores": candidate.get("evidence_scores", {}),
            }
            for candidate in budget_suppressed
        ]
    metadata["merged_candidate_count"] = len(merged)
    return combined_score, metadata, merged


def review_components(
    reference_roi: np.ndarray,
    aligned_roi: np.ndarray,
    valid_roi: np.ndarray | None,
    *,
    reference_size: tuple[int, int],
    roi_area_ratio: float,
    require_whole_cross_evidence: bool,
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    """Choose the same small-ROI or tiled-large-ROI path used by the UI."""
    if roi_area_ratio >= TILE_TRIGGER_ROI_AREA_RATIO:
        return tiled_fused_components(
            reference_roi,
            aligned_roi,
            valid_roi,
            reference_size=reference_size,
            require_whole_cross_evidence=require_whole_cross_evidence,
        )
    score, metadata, candidates = fused_components(
        reference_roi,
        aligned_roi,
        require_cross_evidence=require_whole_cross_evidence,
        valid=valid_roi,
    )
    metadata.update(
        mode="whole_roi_only_small_region",
        roi_size_pixels=[reference_roi.shape[1], reference_roi.shape[0]],
        tile_count=0,
        whole_roi_candidate_count=len(candidates),
        tile_candidate_count=0,
        merged_candidate_count=len(candidates),
    )
    for candidate in candidates:
        candidate.update(
            source_tiles=["whole_roi"],
            evidence_summary={
                "source_tile_count": 0,
                "whole_roi_overlap": True,
                "tile_edge_hits": 0,
                "tile_edge_only": False,
                "merged_observation_count": 1,
            },
        )
    return score, metadata, candidates
