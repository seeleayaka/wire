"""Audit IntelliMan topology evidence against Mendeley YOLO label boxes.

This module is deliberately independent from the cabinet GUI and the DINO/SAM3
review chain.  It consumes an already generated IntelliMan ``topology.json``
and Mendeley source labels in the same image coordinate system.  The labels
are ground truth for evaluation only; they are never used to create a model
prediction or an automatic fault verdict.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class MendeleyIntelliManEvidenceError(ValueError):
    """Raised when Mendeley labels and IntelliMan topology cannot be compared."""


@dataclass(frozen=True)
class MendeleyLabelBox:
    """One source YOLO box, converted into image pixel coordinates."""

    label_index: int
    class_id: int
    bbox_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class IntelliManTopology:
    """The small, geometry-only subset needed for a label-box evidence audit."""

    image_size_width_height: tuple[int, int]
    nodes: tuple[dict[str, Any], ...]
    paths: tuple[dict[str, Any], ...]
    intersections: tuple[dict[str, Any], ...]
    branch_points: tuple[dict[str, Any], ...]


DATASET_IMAGE_LABELS = frozenset({"normal", "damaged", "disconnected", "misrouted"})


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MendeleyIntelliManEvidenceError(f"{name} must be numeric")
    return float(value)


def _integer(value: Any, name: str) -> int:
    number = _number(value, name)
    if not number.is_integer():
        raise MendeleyIntelliManEvidenceError(f"{name} must be an integer")
    return int(number)


def read_image_size(image_path: str | Path) -> tuple[int, int]:
    """Read a Windows Unicode path and return ``(width, height)``."""
    source = Path(image_path)
    raw = np.fromfile(str(source), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise MendeleyIntelliManEvidenceError(f"cannot read image: {source}")
    return int(image.shape[1]), int(image.shape[0])


def render_evidence_overlay(
    image_path: str | Path, report: dict[str, Any], output_path: str | Path
) -> None:
    """Render source boxes for a human review of an evidence report.

    Green means topology geometry was found in or near a source label box;
    orange means none was found.  These are source-label audit markers only,
    not predicted fault classes or inspection verdicts.
    """
    source = Path(image_path)
    raw = np.fromfile(str(source), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise MendeleyIntelliManEvidenceError(f"cannot read image: {source}")
    if report.get("manual_evaluation_only") is not True:
        raise MendeleyIntelliManEvidenceError("evidence overlay requires a manual-evaluation report")
    label_evidence = report.get("label_evidence")
    if not isinstance(label_evidence, list):
        raise MendeleyIntelliManEvidenceError("evidence report lacks label_evidence")

    for item in label_evidence:
        if not isinstance(item, dict):
            raise MendeleyIntelliManEvidenceError("evidence report has an invalid label entry")
        box = item.get("label_bbox_xyxy")
        if not isinstance(box, list) or len(box) != 4:
            raise MendeleyIntelliManEvidenceError("evidence report has an invalid label box")
        left, top, right, bottom = (int(round(_number(value, "label bbox coordinate"))) for value in box)
        evidence_present = item.get("topology_evidence_present")
        if not isinstance(evidence_present, bool):
            raise MendeleyIntelliManEvidenceError("evidence report lacks a topology_evidence_present flag")
        color = (65, 180, 80) if evidence_present else (0, 150, 255)
        cv2.rectangle(image, (left, top), (right, bottom), color, 4)
        label_index = item.get("label_index", "?")
        class_id = item.get("source_class_id", "?")
        state = "topology present" if evidence_present else "no topology"
        text = f"label {label_index} / source class {class_id}: {state}"
        cv2.putText(image, text, (left, max(24, top - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

    output = Path(output_path)
    suffix = output.suffix.lower() or ".png"
    if suffix not in {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}:
        raise MendeleyIntelliManEvidenceError("overlay output must have an image filename extension")
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise MendeleyIntelliManEvidenceError(f"cannot encode overlay image: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(str(output))


def dataset_image_label_from_image_name(image_path: str | Path) -> str | None:
    """Return the Mendeley image-level label encoded by its filename prefix.

    The four image labels are ``normal``, ``damaged``, ``disconnected``, and
    ``misrouted``.  They are distinct from the numeric class IDs stored in the
    individual YOLO box rows.
    """
    prefix = Path(image_path).stem.split("_", 1)[0].lower()
    return prefix if prefix in DATASET_IMAGE_LABELS else None


def fault_family_from_image_name(image_path: str | Path) -> str | None:
    """Compatibility alias for :func:`dataset_image_label_from_image_name`."""
    return dataset_image_label_from_image_name(image_path)


def load_mendeley_yolo_labels(label_path: str | Path, image_size_width_height: tuple[int, int]) -> list[MendeleyLabelBox]:
    """Load raw Mendeley YOLO labels as source evaluation boxes."""
    source = Path(label_path)
    if not source.is_file():
        raise MendeleyIntelliManEvidenceError(f"label file is unavailable: {source}")
    width, height = image_size_width_height
    if width <= 0 or height <= 0:
        raise MendeleyIntelliManEvidenceError("image dimensions must be positive")
    text = source.read_text(encoding="utf-8").strip()
    if not text:
        return []
    labels: list[MendeleyLabelBox] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if len(parts) != 5:
            raise MendeleyIntelliManEvidenceError(f"{source}:{line_number}: expected 5 YOLO values")
        try:
            class_id = _integer(float(parts[0]), f"{source}:{line_number} class id")
            center_x, center_y, box_width, box_height = (float(value) for value in parts[1:])
        except ValueError as error:
            raise MendeleyIntelliManEvidenceError(f"{source}:{line_number}: non-numeric YOLO value") from error
        if not (0.0 <= center_x <= 1.0 and 0.0 <= center_y <= 1.0 and 0.0 < box_width <= 1.0 and 0.0 < box_height <= 1.0):
            raise MendeleyIntelliManEvidenceError(f"{source}:{line_number}: invalid normalized YOLO box")
        left = max(0.0, (center_x - box_width / 2.0) * width)
        top = max(0.0, (center_y - box_height / 2.0) * height)
        right = min(float(width), (center_x + box_width / 2.0) * width)
        bottom = min(float(height), (center_y + box_height / 2.0) * height)
        if right <= left or bottom <= top:
            raise MendeleyIntelliManEvidenceError(f"{source}:{line_number}: empty YOLO box after clipping")
        labels.append(MendeleyLabelBox(len(labels) + 1, class_id, (left, top, right, bottom)))
    return labels


def _records(value: Any, name: str) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise MendeleyIntelliManEvidenceError(f"topology.{name} must be an array of objects")
    return tuple(value)


def _point_row_col(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise MendeleyIntelliManEvidenceError(f"{name} must be [row, col]")
    return _number(value[0], f"{name}[0]"), _number(value[1], f"{name}[1]")


def load_intelliman_topology(topology_path: str | Path) -> IntelliManTopology:
    """Load and validate the coordinate-bearing parts of an IntelliMan report."""
    source = Path(topology_path)
    if not source.is_file():
        raise MendeleyIntelliManEvidenceError(f"IntelliMan topology file is unavailable: {source}")
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise MendeleyIntelliManEvidenceError(f"invalid IntelliMan topology JSON: {source}") from error
    if not isinstance(raw, dict) or raw.get("decision") != "topology_diagnostic_only":
        raise MendeleyIntelliManEvidenceError("topology report is not an IntelliMan diagnostic output")
    input_info = raw.get("input")
    topology = raw.get("topology")
    if not isinstance(input_info, dict) or not isinstance(topology, dict):
        raise MendeleyIntelliManEvidenceError("topology report lacks input or topology section")
    dimensions = input_info.get("image_size_width_height")
    if not isinstance(dimensions, list) or len(dimensions) != 2:
        raise MendeleyIntelliManEvidenceError("topology input image dimensions are unavailable")
    width, height = _integer(dimensions[0], "topology image width"), _integer(dimensions[1], "topology image height")
    if width <= 0 or height <= 0:
        raise MendeleyIntelliManEvidenceError("topology image dimensions must be positive")
    result = IntelliManTopology(
        image_size_width_height=(width, height),
        nodes=_records(topology.get("nodes"), "nodes"),
        paths=_records(topology.get("paths"), "paths"),
        intersections=_records(topology.get("intersections"), "intersections"),
        branch_points=_records(topology.get("branch_points"), "branch_points"),
    )
    for node in result.nodes:
        _point_row_col(node.get("position_row_col"), "topology node position")
    for path in result.paths:
        points = path.get("points_row_col")
        if not isinstance(points, list):
            raise MendeleyIntelliManEvidenceError("topology path has no points_row_col")
        for point in points:
            _point_row_col(point, "topology path point")
    for name, records in (("intersection", result.intersections), ("branch point", result.branch_points)):
        for record in records:
            _point_row_col(record.get("position_row_col"), f"topology {name} position")
    return result


def _expanded_box(box: tuple[float, float, float, float], padding_px: float) -> tuple[float, float, float, float]:
    if padding_px < 0:
        raise MendeleyIntelliManEvidenceError("padding_px must be non-negative")
    left, top, right, bottom = box
    return left - padding_px, top - padding_px, right + padding_px, bottom + padding_px


def _point_in_box(row_col: tuple[float, float], box: tuple[float, float, float, float]) -> bool:
    row, col = row_col
    left, top, right, bottom = box
    return left <= col <= right and top <= row <= bottom


def _segment_intersects_box(
    first_row_col: tuple[float, float], second_row_col: tuple[float, float], box: tuple[float, float, float, float]
) -> bool:
    """Use Liang-Barsky clipping to test a topology polyline segment against a box."""
    first_row, first_col = first_row_col
    second_row, second_col = second_row_col
    left, top, right, bottom = box
    delta_x, delta_y = second_col - first_col, second_row - first_row
    lower, upper = 0.0, 1.0
    for p, q in ((-delta_x, first_col - left), (delta_x, right - first_col), (-delta_y, first_row - top), (delta_y, bottom - first_row)):
        if p == 0:
            if q < 0:
                return False
            continue
        ratio = q / p
        if p < 0:
            if ratio > upper:
                return False
            lower = max(lower, ratio)
        else:
            if ratio < lower:
                return False
            upper = min(upper, ratio)
    return lower <= upper


def _point_records_in_box(records: tuple[dict[str, Any], ...], box: tuple[float, float, float, float]) -> list[str]:
    matched: list[str] = []
    for index, record in enumerate(records, start=1):
        if _point_in_box(_point_row_col(record["position_row_col"], "topology point position"), box):
            identifier = record.get("node_id", record.get("path_id", index))
            matched.append(str(identifier))
    return matched


def evidence_for_label(label: MendeleyLabelBox, topology: IntelliManTopology, padding_px: float = 0.0) -> dict[str, Any]:
    """Return non-decisive topology evidence local to one Mendeley source label."""
    box = _expanded_box(label.bbox_xyxy, padding_px)
    node_ids = _point_records_in_box(topology.nodes, box)
    endpoint_node_ids = [
        str(node.get("node_id", index))
        for index, node in enumerate(topology.nodes, start=1)
        if _point_in_box(_point_row_col(node["position_row_col"], "topology node position"), box)
        and _integer(node.get("degree", 0), "topology node degree") <= 1
    ]
    path_ids: list[str] = []
    segment_hits = 0
    for index, path in enumerate(topology.paths, start=1):
        points = [_point_row_col(point, "topology path point") for point in path["points_row_col"]]
        hit_count = sum(_segment_intersects_box(first, second, box) for first, second in zip(points, points[1:]))
        if hit_count:
            path_ids.append(str(path.get("path_id", index)))
            segment_hits += hit_count
    intersection_ids = _point_records_in_box(topology.intersections, box)
    branch_point_ids = _point_records_in_box(topology.branch_points, box)
    evidence_present = bool(node_ids or path_ids or intersection_ids or branch_point_ids)
    return {
        "label_index": label.label_index,
        "source_class_id": label.class_id,
        "label_bbox_xyxy": [round(value, 3) for value in label.bbox_xyxy],
        "padding_px": padding_px,
        "topology_node_ids_in_or_near_box": node_ids,
        "topology_endpoint_node_ids_in_or_near_box": endpoint_node_ids,
        "topology_path_ids_intersecting_or_near_box": path_ids,
        "topology_path_segment_hit_count": segment_hits,
        "topology_intersection_ids_in_or_near_box": intersection_ids,
        "topology_branch_point_ids_in_or_near_box": branch_point_ids,
        "topology_evidence_present": evidence_present,
        "evidence_state": (
            "topology_evidence_near_source_label_manual_evaluation"
            if evidence_present
            else "no_topology_evidence_near_source_label_manual_evaluation"
        ),
    }


def evaluate_mendeley_intelliman_evidence(
    image_path: str | Path, label_path: str | Path, topology_path: str | Path, padding_px: float = 0.0
) -> dict[str, Any]:
    """Compare an existing IntelliMan topology result with Mendeley source boxes."""
    image_size = read_image_size(image_path)
    labels = load_mendeley_yolo_labels(label_path, image_size)
    topology = load_intelliman_topology(topology_path)
    if topology.image_size_width_height != image_size:
        raise MendeleyIntelliManEvidenceError(
            "topology image dimensions do not match the Mendeley image; do not compare different coordinate systems"
        )
    evidence = [evidence_for_label(label, topology, padding_px) for label in labels]
    return {
        "schema_version": 1,
        "purpose": "Mendeley image-label and source-box to IntelliMan topology evidence audit; evaluation only.",
        "manual_evaluation_only": True,
        "coordinate_contract": "Mendeley image, YOLO labels, and IntelliMan topology.json must share exact image dimensions.",
        "source": {
            "image": str(Path(image_path).resolve()),
            "label": str(Path(label_path).resolve()),
            "intelliman_topology": str(Path(topology_path).resolve()),
            "image_size_width_height": list(image_size),
            "dataset_image_label": dataset_image_label_from_image_name(image_path),
            "fault_family_from_filename": fault_family_from_image_name(image_path),
        },
        "rules": {
            "topology_evidence_padding_px": padding_px,
            "dataset_image_label_mapping": "filename prefix: normal, damaged, disconnected, or misrouted",
            "source_box_class_mapping": "numeric YOLO box IDs preserved without inferred semantics",
        },
        "summary": {
            "source_label_count": len(evidence),
            "labels_with_topology_evidence": sum(item["topology_evidence_present"] for item in evidence),
            "labels_without_topology_evidence": sum(not item["topology_evidence_present"] for item in evidence),
            "topology_node_count": len(topology.nodes),
            "topology_path_count": len(topology.paths),
        },
        "label_evidence": evidence,
        "not_claimed": [
            "model_fault_prediction",
            "automatic_fault_classification",
            "cable_identity",
            "terminal_assignment",
            "electrical_continuity",
            "production_acceptance_or_rejection",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, help="optional source-label evidence overlay for manual review")
    parser.add_argument("--padding-px", type=float, default=0.0)
    args = parser.parse_args()
    report = evaluate_mendeley_intelliman_evidence(args.image, args.labels, args.topology, args.padding_px)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.overlay is not None:
        render_evidence_overlay(args.image, report, args.overlay)
    print(json.dumps(report["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
