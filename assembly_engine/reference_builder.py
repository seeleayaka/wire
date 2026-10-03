"""Automatic candidate extraction from one approved cabinet reference image.

This is intentionally a candidate builder, not a material-number recognizer.
It emits only stable internal IDs and records every rejected visual segment in
the build report so downstream decisions remain auditable.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .assembly_template import AssemblyTemplate, TemplateObject, TemplateRule


@dataclass(frozen=True)
class Candidate:
    center: tuple[float, float]
    first: tuple[int, int]
    second: tuple[int, int]
    direction_deg: float
    region: tuple[int, int, int, int]
    color_bgr: tuple[float, float, float]
    color_name: str
    confidence: float
    source: str
    metrics: dict[str, float]


def read_image(path: str | Path) -> np.ndarray:
    """Read Unicode Windows paths without relying on cv2.imread."""
    source = Path(path)
    image = cv2.imdecode(np.fromfile(str(source), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read reference image: {source}")
    return image


def write_image(path: str | Path, image: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(target.suffix or ".png", image)
    if not ok:
        raise ValueError(f"cannot encode image: {target}")
    encoded.tofile(str(target))


def _color_name(bgr: tuple[float, float, float]) -> str:
    sample = np.uint8([[np.clip(np.round(bgr), 0, 255)]])
    hue, saturation, value = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)[0, 0]
    if value < 70:
        return "black"
    if saturation < 45:
        return "white" if value > 160 else "gray"
    if hue < 8 or hue >= 172:
        return "red"
    if hue < 22:
        return "orange"
    if hue < 38:
        return "yellow"
    if hue < 86:
        return "green"
    if hue < 112:
        return "cyan"
    if hue < 142:
        return "blue"
    return "purple"


def _angle_degrees(first: tuple[int, int], second: tuple[int, int]) -> float:
    angle = np.degrees(np.arctan2(second[1] - first[1], second[0] - first[0]))
    return float(angle % 180.0)


def _region_from_line(first: tuple[int, int], second: tuple[int, int], width: int, height: int) -> tuple[int, int, int, int]:
    pad = max(10, min(width, height) // 90)
    left = max(0, min(first[0], second[0]) - pad)
    top = max(0, min(first[1], second[1]) - pad)
    right = min(width, max(first[0], second[0]) + pad + 1)
    bottom = min(height, max(first[1], second[1]) + pad + 1)
    return left, top, right, bottom


def _line_support(image: np.ndarray, first: tuple[int, int], second: tuple[int, int]) -> tuple[float, tuple[float, float, float]]:
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.line(mask, first, second, 255, thickness=7)
    pixels = image[mask > 0]
    if pixels.size == 0:
        return 0.0, (0.0, 0.0, 0.0)
    hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    color_or_white = np.logical_or(hsv[:, 1] >= 55, hsv[:, 2] >= 150)
    return float(np.mean(color_or_white)), tuple(float(item) for item in np.median(pixels, axis=0))


def _too_close_to_frame(region: tuple[int, int, int, int], width: int, height: int) -> bool:
    left, top, right, bottom = region
    margin_x, margin_y = max(4, width // 80), max(4, height // 80)
    return left <= margin_x or top <= margin_y or right >= width - margin_x or bottom >= height - margin_y


def _overlap(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    smaller = min((first[2] - first[0]) * (first[3] - first[1]), (second[2] - second[0]) * (second[3] - second[1]))
    return intersection / max(smaller, 1)


def _deduplicate(candidates: list[Candidate]) -> list[Candidate]:
    accepted: list[Candidate] = []
    for candidate in sorted(candidates, key=lambda item: item.confidence, reverse=True):
        if all(_overlap(candidate.region, kept.region) < 0.72 for kept in accepted):
            accepted.append(candidate)
    return sorted(accepted, key=lambda item: (item.center[1], item.center[0]))


def extract_wire_candidates(image: np.ndarray, max_candidates: int = 32) -> tuple[list[Candidate], list[dict[str, Any]]]:
    """Find line-like wire segments while rejecting obvious frame and rail edges."""
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 45, 125)
    minimum_length = max(26, round(min(width, height) * 0.055))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=max(22, minimum_length // 2), minLineLength=minimum_length, maxLineGap=11)
    accepted: list[Candidate] = []
    rejected: list[dict[str, Any]] = []
    if lines is None:
        return accepted, rejected
    for raw in lines.reshape(-1, 4):
        x1, y1, x2, y2 = (int(value) for value in raw)
        length = float(np.hypot(x2 - x1, y2 - y1))
        angle = _angle_degrees((x1, y1), (x2, y2))
        region = _region_from_line((x1, y1), (x2, y2), width, height)
        reason: str | None = None
        horizontal = min(abs(angle), abs(180.0 - angle)) <= 12.0
        # In this cabinet view the dominant long horizontal edges are rails,
        # terminal rows, and label/slot boundaries rather than cable runs.
        # Rejecting them early keeps the automatically generated template from
        # turning every terminal-row edge into a pseudo-wire.  Diagonal and
        # near-vertical candidates remain eligible for the first version.
        if horizontal:
            reason = "horizontal_visual_structure"
        elif _too_close_to_frame(region, width, height):
            reason = "near_image_frame"
        elif length >= max(width, height) * 0.68:
            reason = "cabinet_scale_edge"
        support, color_bgr = _line_support(image, (x1, y1), (x2, y2))
        if reason is None and support < 0.42:
            reason = "weak_wire_appearance"
        if reason is not None:
            rejected.append({"region": list(region), "direction_deg": round(angle, 2), "reason": reason})
            continue
        aspect_signal = min(1.0, length / max(minimum_length * 2.0, 1.0))
        color_signal = max(0.0, min(1.0, (support - 0.35) / 0.65))
        confidence = round(0.45 * aspect_signal + 0.55 * color_signal, 3)
        if confidence < 0.56:
            rejected.append({"region": list(region), "direction_deg": round(angle, 2), "reason": "candidate_confidence_too_low", "confidence": confidence})
            continue
        accepted.append(
            Candidate(
                center=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                first=(x1, y1),
                second=(x2, y2),
                direction_deg=angle,
                region=region,
                color_bgr=color_bgr,
                color_name=_color_name(color_bgr),
                confidence=confidence,
                source="hough_line_with_wire_appearance",
                metrics={"length_px": round(length, 2), "appearance_support": round(support, 3)},
            )
        )
    candidates = _deduplicate(accepted)[:max_candidates]
    return candidates, rejected


def _to_template_object(candidate: Candidate, number: int, width: int, height: int) -> TemplateObject:
    left, top, right, bottom = candidate.region
    bgr = [round(component, 2) for component in candidate.color_bgr]
    tolerance = 62.0 if candidate.color_name in {"white", "gray", "black"} else 48.0
    return TemplateObject(
        id=f"wire_{number:03d}",
        kind="wire_candidate",
        color={"name": candidate.color_name, "bgr": bgr, "tolerance": tolerance},
        center_norm=[round(candidate.center[0] / width, 6), round(candidate.center[1] / height, 6)],
        direction_deg=round(candidate.direction_deg, 3),
        expected_region=[round(left / width, 6), round(top / height, 6), round(right / width, 6), round(bottom / height, 6)],
        presence_threshold=0.60,
        position_tolerance_px=max(12.0, round(min(right - left, bottom - top) * 0.55, 2)),
        direction_tolerance_deg=20.0,
        confidence="automatic_candidate",
        candidate_confidence=candidate.confidence,
        reference_metrics=candidate.metrics,
        reference_line_norm=[
            round(candidate.first[0] / width, 6),
            round(candidate.first[1] / height, 6),
            round(candidate.second[0] / width, 6),
            round(candidate.second[1] / height, 6),
        ],
    )


def build_reference_template(
    image: np.ndarray,
    reference_image: str,
    template_id: str = "current_cable_cabinet_v1",
) -> tuple[AssemblyTemplate, np.ndarray, dict[str, Any]]:
    """Build one single-reference template plus overlay and audit report."""
    height, width = image.shape[:2]
    candidates, rejected = extract_wire_candidates(image)
    objects = [_to_template_object(candidate, index, width, height) for index, candidate in enumerate(candidates, 1)]
    rules: list[TemplateRule] = []
    for item in objects:
        rules.extend(
            [
                TemplateRule(id=f"{item.id}_presence", type="exists", object_id=item.id),
                TemplateRule(id=f"{item.id}_color", type="color", object_id=item.id),
                TemplateRule(id=f"{item.id}_position", type="position", object_id=item.id),
            ]
        )
    overlay = image.copy()
    for item in objects:
        left, top, right, bottom = (round(value * scale) for value, scale in zip(item.expected_region, (width, height, width, height)))
        cv2.rectangle(overlay, (left, top), (right, bottom), (0, 215, 255), 2)
        cv2.putText(overlay, item.id, (left, max(18, top - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 215, 255), 1, cv2.LINE_AA)
    report = {
        "template_id": template_id,
        "reference_image": reference_image,
        "image_size": [width, height],
        "candidate_count": len(objects),
        "candidate_sources": {"hough_line_with_wire_appearance": len(objects)},
        "rejected_count": len(rejected),
        "rejected_candidates": rejected,
        "connector_candidates": [],
        "warnings": [
            "Connector candidates are retained for the future topology stage and are not generated as first-version rules.",
            "All objects are automatic visual candidates, not confirmed material or terminal identifiers.",
        ],
    }
    template = AssemblyTemplate(
        template_id=template_id,
        reference_image=reference_image,
        image_size=[width, height],
        build_mode="automatic_single_reference",
        objects=objects,
        rules=rules,
        metadata={"build_report": {"candidate_count": len(objects), "rejected_count": len(rejected)}},
    )
    return template, overlay, report
