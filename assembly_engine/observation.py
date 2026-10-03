"""Extract actual visual evidence from an inspection image already aligned to a template."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .assembly_template import AssemblyTemplate, TemplateObject


def _pixels(image: np.ndarray, region: list[float]) -> tuple[int, int, int, int]:
    height, width = image.shape[:2]
    left, top, right, bottom = region
    return round(left * width), round(top * height), round(right * width), round(bottom * height)


def _lab_distance(expected_bgr: list[float], actual_bgr: np.ndarray) -> float:
    expected = np.uint8([[np.clip(np.round(expected_bgr), 0, 255)]])
    actual = np.uint8([[np.clip(np.round(actual_bgr), 0, 255)]])
    expected_lab = cv2.cvtColor(expected, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
    actual_lab = cv2.cvtColor(actual, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
    return float(np.linalg.norm(expected_lab - actual_lab))


def _line_candidate(
    crop: np.ndarray,
    expected_direction: float,
    expected_center: tuple[float, float] | None = None,
) -> tuple[tuple[float, float] | None, float | None, float, float]:
    # Use the strongest channel rather than luminance alone.  A blue cable is
    # visually high-contrast against a dark cabinet but has low luminance; a
    # grayscale conversion would erase its line evidence and make a wrong
    # color look like a missing object.
    gray = np.max(crop, axis=2).astype(np.uint8)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 45, 125)
    minimum = max(14, min(crop.shape[:2]) // 3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=max(12, minimum // 2), minLineLength=minimum, maxLineGap=7)
    if lines is None:
        return None, None, 0.0, float(np.mean(edges > 0))
    best: tuple[float, tuple[float, float], float] | None = None
    for raw in lines.reshape(-1, 4):
        x1, y1, x2, y2 = (int(value) for value in raw)
        length = float(np.hypot(x2 - x1, y2 - y1))
        direction = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0)
        angle_error = min(abs(direction - expected_direction), 180.0 - abs(direction - expected_direction))
        direction_score = max(0.0, 1.0 - angle_error / 90.0)
        midpoint = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        if expected_center is None:
            center_score = 1.0
        else:
            distance = float(np.hypot(midpoint[0] - expected_center[0], midpoint[1] - expected_center[1]))
            center_score = max(0.0, 1.0 - distance / max(float(np.hypot(*crop.shape[:2])) * 0.65, 1.0))
        score = length * direction_score * (0.55 + 0.45 * center_score)
        if best is None or score > best[0]:
            best = (score, ((x1 + x2) / 2.0, (y1 + y2) / 2.0), direction)
    if best is None:
        return None, None, 0.0, float(np.mean(edges > 0))
    max_length = float(np.hypot(crop.shape[1], crop.shape[0]))
    return best[1], best[2], min(1.0, best[0] / max(max_length, 1.0)), float(np.mean(edges > 0))


def _line_band(crop: np.ndarray, center: tuple[float, float], direction_deg: float) -> np.ndarray:
    """Return a mask for a narrow band around the detected line.

    The template region usually contains much more cabinet/background than
    wire pixels.  Sampling the whole region therefore makes a correct colored
    wire look black/gray.  A band around the detected line keeps color evidence
    local without requiring a material classifier.
    """
    height, width = crop.shape[:2]
    theta = np.radians(direction_deg)
    half_length = float(np.hypot(width, height))
    dx, dy = np.cos(theta) * half_length, np.sin(theta) * half_length
    first = (round(center[0] - dx), round(center[1] - dy))
    second = (round(center[0] + dx), round(center[1] + dy))
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.line(mask, first, second, 255, thickness=max(3, min(9, round(min(height, width) * 0.08))))
    return mask


def _sample_line_color(
    crop: np.ndarray,
    center: tuple[float, float] | None,
    direction_deg: float | None,
    expected_bgr: list[float],
    tolerance: float,
) -> np.ndarray:
    """Estimate color from the detected line, preferring pixels near expected color."""
    if center is None or direction_deg is None:
        return np.median(crop.reshape(-1, 3), axis=0)
    band = _line_band(crop, center, direction_deg)
    pixels = crop[band > 0]
    if pixels.size == 0:
        return np.median(crop.reshape(-1, 3), axis=0)

    # When the expected color is present (the normal/reference case), use only
    # close pixels so black cabinet/background does not dominate the median.
    expected = np.uint8([[np.clip(np.round(expected_bgr), 0, 255)]])
    expected_lab = cv2.cvtColor(expected, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
    lab_pixels = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    distances = np.linalg.norm(lab_pixels - expected_lab, axis=1)
    close = pixels[distances <= max(24.0, float(tolerance) * 1.35)]
    if close.shape[0] >= max(5, pixels.shape[0] // 12):
        return np.median(close, axis=0)
    # If the color really changed, retain the detected line pixels so the rule
    # engine can report a wrong-color suspicion instead of sampling background.
    return np.median(pixels, axis=0)


def _structural_similarity(reference: np.ndarray, inspection: np.ndarray) -> tuple[float, float]:
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    inspection_gray = cv2.cvtColor(inspection, cv2.COLOR_BGR2GRAY)
    reference_gray = cv2.GaussianBlur(reference_gray, (5, 5), 0)
    inspection_gray = cv2.GaussianBlur(inspection_gray, (5, 5), 0)
    difference = cv2.absdiff(reference_gray, inspection_gray)
    normalized_difference = float(np.mean(difference)) / 255.0
    return max(0.0, 1.0 - normalized_difference * 2.2), normalized_difference


def observe_object(
    reference: np.ndarray,
    aligned_inspection: np.ndarray,
    item: TemplateObject,
    external_evidence: dict[str, Any] | None = None,
    reference_self_check: bool = False,
) -> dict[str, Any]:
    """Observe a candidate object only inside its template-defined region."""
    left, top, right, bottom = _pixels(reference, item.expected_region)
    reference_crop = reference[top:bottom, left:right]
    inspection_crop = aligned_inspection[top:bottom, left:right]
    if reference_crop.size == 0 or inspection_crop.size == 0 or reference_crop.shape != inspection_crop.shape:
        return {"object_id": item.id, "observable": False, "reason": "invalid_or_outside_expected_region"}
    expected_center_x = item.center_norm[0] * reference.shape[1]
    expected_center_y = item.center_norm[1] * reference.shape[0]
    expected_center_local = (expected_center_x - left, expected_center_y - top)
    actual_center, actual_direction, line_strength, edge_density = _line_candidate(
        inspection_crop,
        item.direction_deg,
        expected_center_local,
    )
    structural_similarity, traditional_difference = _structural_similarity(reference_crop, inspection_crop)
    median_bgr = _sample_line_color(
        inspection_crop,
        actual_center,
        actual_direction,
        item.color["bgr"],
        float(item.color["tolerance"]),
    )
    color_distance = _lab_distance(item.color["bgr"], median_bgr)
    color_similarity = max(0.0, 1.0 - color_distance / max(float(item.color["tolerance"]), 1.0))
    reference_edge = float(item.reference_metrics.get("appearance_support", item.reference_metrics.get("edge_density", 0.20)))
    expected_edge_support = max(0.08, min(1.0, reference_edge if reference_edge <= 1.0 else reference_edge / 255.0))
    edge_similarity = min(1.0, edge_density / expected_edge_support)
    presence_score = round(0.48 * structural_similarity + 0.32 * line_strength + 0.12 * color_similarity + 0.08 * edge_similarity, 4)
    observation: dict[str, Any] = {
        "object_id": item.id,
        "observable": True,
        "expected_region": item.expected_region,
        "actual_color_bgr": [round(float(value), 2) for value in median_bgr],
        "color_distance_lab": round(color_distance, 3),
        "color_similarity": round(color_similarity, 4),
        "structural_similarity": round(structural_similarity, 4),
        "traditional_difference": round(traditional_difference, 4),
        "line_strength": round(line_strength, 4),
        "edge_density": round(edge_density, 5),
        "presence_score": presence_score,
        "actual_center_norm": None,
        "actual_direction_deg": None if actual_direction is None else round(actual_direction, 3),
        "position_error_px": None,
        "direction_error_deg": None,
        "evidence": {"external": external_evidence or {}},
        "reference_self_check": reference_self_check,
    }
    if actual_center is not None and actual_direction is not None:
        actual_x, actual_y = actual_center[0] + left, actual_center[1] + top
        direction_error = min(abs(actual_direction - item.direction_deg), 180.0 - abs(actual_direction - item.direction_deg))
        observation.update(
            actual_center_norm=[round(actual_x / reference.shape[1], 6), round(actual_y / reference.shape[0], 6)],
            position_error_px=round(float(np.hypot(actual_x - expected_center_x, actual_y - expected_center_y)), 3),
            direction_error_deg=round(direction_error, 3),
        )
    return observation


def observe_template(
    reference: np.ndarray,
    aligned_inspection: np.ndarray,
    template: AssemblyTemplate,
    external_evidence: dict[str, dict[str, Any]] | None = None,
    reference_self_check: bool = False,
) -> dict[str, Any]:
    """Produce actual observations without altering the immutable template."""
    if [reference.shape[1], reference.shape[0]] != template.image_size:
        raise ValueError(
            f"reference size {reference.shape[1]}x{reference.shape[0]} does not match template "
            f"{template.image_size[0]}x{template.image_size[1]}"
        )
    if aligned_inspection.shape[:2] != reference.shape[:2]:
        raise ValueError("inspection image must already be aligned to reference dimensions")
    evidence = external_evidence or {}
    observations = [
        observe_object(reference, aligned_inspection, item, evidence.get(item.id), reference_self_check)
        for item in template.objects
    ]
    return {
        "template_id": template.template_id,
        "image_size": template.image_size,
        "observation_count": len(observations),
        "actual_observations": observations,
    }
