"""Image-bound, operator-assisted port maps and conservative topology review.

Body inventories are not ports. This optional boundary wrapper does not alter
the existing SAM adapter or graph comparator, and never issues a fault verdict.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from pathlib import Path

from .sam_topology_adapter import connection_graph_from_sam_endpoints
from .topology import ConnectionGraph, compare_topologies


def _text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be nonempty text")
    return value.strip()


def _box(value, size):
    if not isinstance(value, list) or len(value) != 4 or any(
        isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) for x in value
    ):
        raise ValueError("port ROI must contain four finite numbers")
    x1, y1, x2, y2 = value
    if not (0 <= x1 < x2 <= size[0] and 0 <= y1 < y2 <= size[1]):
        raise ValueError("port ROI outside its image or inverted")


def image_binding(image_path: Path) -> dict:
    from PIL import Image
    with Image.open(image_path) as image:
        size = list(image.size)
        image.verify()
    return {"image_path": str(image_path.resolve()),
            "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            "image_size": size, "coordinate_frame": "source_image_pixels"}


def create_mapping_draft(image_path: Path, scene_type: str, inventory: dict | None = None) -> dict:
    binding = image_binding(image_path)
    objects = []
    if inventory is not None:
        if inventory.get("source", {}).get("image_sha256") != binding["image_sha256"] or inventory.get("image_size") != binding["image_size"]:
            raise ValueError("body inventory is not bound to this image")
        objects = deepcopy(inventory.get("objects", []))
    return {"schema_version": 1, "map_id": image_path.stem + "_port_map",
            "scene_type": _text(scene_type, "scene_type"), "image_binding": binding,
            "body_objects": objects, "ports": [], "expected_connections": None,
            "expected_review": {"confirmed": False, "reviewer": None, "evidence_note": None},
            "scope": {"description": "Pending operator-defined visible-port scope",
                      "expected_complete": False},
            "claim_boundary": "Port map draft, not automatic identification or electrical continuity."}


def validate_mapping(mapping: dict, image_path: Path) -> dict:
    if not isinstance(mapping, dict) or type(mapping.get("schema_version")) is not int or mapping["schema_version"] != 1:
        raise ValueError("unsupported port mapping schema")
    _text(mapping.get("map_id"), "map_id")
    _text(mapping.get("scene_type"), "scene_type")
    actual = image_binding(image_path)
    bound = mapping.get("image_binding")
    if not isinstance(bound, dict) or any(bound.get(k) != actual[k] for k in ("image_sha256", "image_size", "coordinate_frame")):
        raise ValueError("port map image hash, size or coordinate frame mismatch; no automatic reuse of coordinates")
    scope = mapping.get("scope")
    if not isinstance(scope, dict) or type(scope.get("expected_complete")) is not bool:
        raise ValueError("scope completeness must be explicitly declared")
    _text(scope.get("description"), "scope.description")
    ports = mapping.get("ports")
    if not isinstance(ports, list):
        raise ValueError("ports must be a list")
    ids = set()
    for port in ports:
        if not isinstance(port, dict):
            raise ValueError("port must be an object")
        port_id = _text(port.get("id"), "port.id")
        if port_id in ids:
            raise ValueError("duplicate port ID")
        ids.add(port_id)
        _text(port.get("device_id"), "port.device_id")
        _text(port.get("terminal_label"), "port.terminal_label")
        if port.get("roi_kind") != "wire_entry_port":
            raise ValueError("only wire-entry port ROIs may enter topology; body boxes are not ports")
        _box(port.get("bbox_xyxy"), actual["image_size"])
        if type(port.get("confirmed")) is not bool:
            raise ValueError("port confirmation must be explicit boolean")
        if port["confirmed"]:
            _text(port.get("reviewer"), "port.reviewer")
            _text(port.get("evidence_note"), "port.evidence_note")
    expected = mapping.get("expected_connections")
    if expected is not None and not isinstance(expected, list):
        raise ValueError("expected_connections must be null (unknown) or a declared list")
    review = mapping.get("expected_review")
    if not isinstance(review, dict) or type(review.get("confirmed")) is not bool:
        raise ValueError("expected review must be explicit")
    if review["confirmed"]:
        if expected is None:
            raise ValueError("unknown expected connections cannot be confirmed")
        _text(review.get("reviewer"), "expected reviewer")
        _text(review.get("evidence_note"), "expected evidence_note")
    if expected is not None:
        ConnectionGraph.from_dict({"schema_version": 1, "graph_id": mapping["map_id"] + "_expected",
            "scene_type": mapping["scene_type"], "nodes": [{"id": p["id"]} for p in ports], "connections": expected})
    return actual


def review_mapped_topology(mapping: dict, image_path: Path, endpoint_report: dict | None = None) -> dict:
    """Evaluate only explicitly mapped ports, retaining missing-evidence guards.

The fixed score gate stays .75. Coverage is a separate operator declaration,
never inferred from an empty SAM result. Agreement remains a review proposal.
"""
    binding = validate_mapping(mapping, image_path)
    ports = mapping["ports"]
    reasons = []
    if not ports:
        reasons.append("no_wire_entry_port_rois")
    elif any(not p["confirmed"] for p in ports):
        reasons.append("port_identity_or_roi_unconfirmed")
    if mapping["expected_connections"] is None:
        reasons.append("expected_connections_unknown")
    if not mapping["expected_review"]["confirmed"]:
        reasons.append("expected_connections_unreviewed")
    if not mapping["scope"]["expected_complete"]:
        reasons.append("expected_scope_incomplete")
    result = {"schema_version": 1, "map_id": mapping["map_id"], "image_binding": binding,
        "decision": "insufficient_evidence", "automatic_fault_verdict": False,
        "manual_confirmation_required": True, "minimum_confidence": .75,
        "reasons": reasons, "adapter_audit": None, "observed_graph": None, "raw_comparison": None,
        "port_count": len(ports), "confirmed_port_count": sum(p["confirmed"] for p in ports),
        "claim_boundary": "Visible segment relation proposals only; no hidden path or electrical continuity."}
    if endpoint_report is None:
        reasons.append("endpoint_evidence_not_supplied")
        return result
    if not isinstance(endpoint_report, dict):
        raise ValueError("endpoint report must be an object")
    endpoint_binding = endpoint_report.get("image_binding")
    if not isinstance(endpoint_binding, dict) or any(endpoint_binding.get(k) != binding[k] for k in ("image_sha256", "image_size", "coordinate_frame")):
        raise ValueError("endpoint report is not bound to the mapped image coordinate frame")
    records = endpoint_report.get("records")
    if not isinstance(records, list):
        raise ValueError("endpoint records must be a list")
    if not records:
        reasons.append("no_visible_segment_records")
    if endpoint_report.get("geometry_contract_version") != 1 or type(endpoint_report.get("geometry_contract_version")) is not int:
        reasons.append("endpoint_geometry_contract_unverified")
        return result
    # Reject nonfinite/out-of-frame endpoints before invoking the legacy adapter.
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("endpoint record must be an object")
        score = record.get("source_score")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("endpoint score must be finite within 0..1")
        ends = record.get("visible_ends_xy", [])
        if not isinstance(ends, list):
            raise ValueError("visible ends must be a list")
        for point in ends:
            if not isinstance(point, list) or len(point) != 2 or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in point):
                raise ValueError("endpoint coordinates must be finite")
            if not (0 <= point[0] < binding["image_size"][0] and 0 <= point[1] < binding["image_size"][1]):
                raise ValueError("endpoint coordinates outside mapped image")
    coverage = endpoint_report.get("coverage_review", {})
    if not isinstance(coverage, dict) or coverage.get("complete") is not True:
        reasons.append("observed_scope_coverage_unconfirmed")
    else:
        _text(coverage.get("reviewer"), "coverage reviewer")
        _text(coverage.get("evidence_note"), "coverage evidence_note")
        if coverage.get("map_id") != mapping["map_id"]:
            raise ValueError("coverage review references a different map")
    # Draft ports must never generate edge candidates.
    if not ports or any(not p["confirmed"] for p in ports):
        return result
    terminal_map = {"schema_version": 1, "graph_id": mapping["map_id"] + "_observed",
        "scene_type": mapping["scene_type"], "terminals": [deepcopy(p) for p in ports]}
    observed, audit = connection_graph_from_sam_endpoints(endpoint_report, terminal_map)
    result["observed_graph"], result["adapter_audit"] = observed.to_dict(), audit
    if audit["summary"]["unresolved_record_count"]:
        reasons.append("unresolved_visible_segment_endpoints")
    if mapping["expected_connections"] is None or not mapping["expected_review"]["confirmed"]:
        return result
    expected = ConnectionGraph.from_dict({"schema_version": 1, "graph_id": mapping["map_id"] + "_expected",
        "scene_type": mapping["scene_type"], "nodes": [{"id": p["id"]} for p in ports],
        "connections": mapping["expected_connections"]})
    comparison = compare_topologies(expected, observed, minimum_confidence=.75)
    result["raw_comparison"] = comparison
    if comparison["decision"] == "insufficient_evidence":
        reasons.append("low_score_connection_evidence")
    if not reasons:
        result["decision"] = ("declared_visible_topology_agreement_manual_review"
            if comparison["decision"] == "match" else comparison["decision"])
    return result
