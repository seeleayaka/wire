"""Conservative bridge from SAM3 visible-segment endpoints to a declared graph.

This module does not infer cable identity or electrical continuity.  It only
creates a candidate graph edge when both ends of one independent SAM3 segment
map uniquely to two different operator-declared terminal regions.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .topology import ConnectionGraph, TopologyValidationError


class SamTopologyAdapterError(TopologyValidationError):
    """Raised when the SAM endpoint or terminal-map contract is invalid."""


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SamTopologyAdapterError(f"{field} must be a non-empty string")
    return value.strip()


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SamTopologyAdapterError(f"{field} must be numeric")
    return float(value)


def _terminal_nodes(terminal_map: dict[str, Any]) -> tuple[str, str, list[dict[str, Any]]]:
    if not isinstance(terminal_map, dict) or terminal_map.get("schema_version") != 1:
        raise SamTopologyAdapterError("terminal map schema_version must be 1")
    graph_id = _text(terminal_map.get("graph_id"), "terminal map graph_id")
    scene_type = _text(terminal_map.get("scene_type"), "terminal map scene_type")
    raw_terminals = terminal_map.get("terminals")
    if not isinstance(raw_terminals, list):
        raise SamTopologyAdapterError("terminal map terminals must be a list")

    result: list[dict[str, Any]] = []
    terminal_ids: set[str] = set()
    for index, raw in enumerate(raw_terminals, start=1):
        if not isinstance(raw, dict):
            raise SamTopologyAdapterError(f"terminal {index} must be an object")
        terminal_id = _text(raw.get("id"), f"terminal {index} id")
        if terminal_id in terminal_ids:
            raise SamTopologyAdapterError(f"duplicate terminal id: {terminal_id}")
        terminal_ids.add(terminal_id)
        bbox = raw.get("bbox_xyxy")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise SamTopologyAdapterError(f"terminal {terminal_id} bbox_xyxy must contain 4 numbers")
        x1, y1, x2, y2 = (
            _number(item, f"terminal {terminal_id} bbox_xyxy") for item in bbox
        )
        if x2 <= x1 or y2 <= y1:
            raise SamTopologyAdapterError(f"terminal {terminal_id} bbox_xyxy must have positive area")
        node = dict(raw)
        node["id"] = terminal_id
        node["bbox_xyxy"] = [x1, y1, x2, y2]
        node.setdefault("node_type", "declared_terminal_roi")
        result.append(node)
    return graph_id, scene_type, result


def _endpoint(point: Any, field: str) -> tuple[float, float]:
    if not isinstance(point, list) or len(point) != 2:
        raise SamTopologyAdapterError(f"{field} must contain x and y")
    return _number(point[0], field), _number(point[1], field)


def _matches(
    point: tuple[float, float], terminals: list[dict[str, Any]], padding: float
) -> list[str]:
    x, y = point
    result: list[str] = []
    for terminal in terminals:
        x1, y1, x2, y2 = terminal["bbox_xyxy"]
        if x1 - padding <= x <= x2 + padding and y1 - padding <= y <= y2 + padding:
            result.append(terminal["id"])
    return sorted(result)


def connection_graph_from_sam_endpoints(
    endpoint_report: dict[str, Any],
    terminal_map: dict[str, Any],
    *,
    endpoint_padding_px: float = 0,
) -> tuple[ConnectionGraph, dict[str, Any]]:
    """Map independent SAM3 visible segments to conservative candidate edges."""
    padding = _number(endpoint_padding_px, "endpoint_padding_px")
    if padding < 0:
        raise SamTopologyAdapterError("endpoint_padding_px must be non-negative")
    if not isinstance(endpoint_report, dict):
        raise SamTopologyAdapterError("endpoint report must be an object")
    raw_records = endpoint_report.get("records")
    if not isinstance(raw_records, list):
        raise SamTopologyAdapterError("endpoint report records must be a list")
    graph_id, scene_type, terminals = _terminal_nodes(terminal_map)

    evidence_records: list[dict[str, Any]] = []
    edges: dict[tuple[str, str], list[tuple[str, float]]] = defaultdict(list)
    record_ids: set[str] = set()

    for index, raw in enumerate(raw_records, start=1):
        if not isinstance(raw, dict):
            raise SamTopologyAdapterError(f"endpoint record {index} must be an object")
        record_id = _text(raw.get("record_id"), f"endpoint record {index} record_id")
        if record_id in record_ids:
            raise SamTopologyAdapterError(f"duplicate endpoint record id: {record_id}")
        record_ids.add(record_id)
        score_value = raw.get("source_score")
        score = 0.0 if score_value is None else _number(score_value, f"{record_id} source_score")
        if not 0.0 <= score <= 1.0:
            raise SamTopologyAdapterError(f"{record_id} source_score must be within 0..1")
        raw_ends = raw.get("visible_ends_xy")
        item: dict[str, Any] = {
            "record_id": record_id,
            "source_score": score,
            "terminal_matches": [],
        }
        # Older extractors collapsed a multi-tip mask to its farthest pair.
        # Reject that lost ambiguity even if visible_ends_xy contains two points.
        if "candidate_tip_count" in raw and raw["candidate_tip_count"] != 2:
            item["state"] = "ambiguous_visible_segment_end_count"
            item["candidate_tip_count"] = raw["candidate_tip_count"]
            evidence_records.append(item)
            continue
        if raw.get("branch_cluster_count", 0) != 0:
            item["state"] = "ambiguous_branched_visible_segment"
            evidence_records.append(item)
            continue
        contract = raw.get("geometry_contract_version", endpoint_report.get("geometry_contract_version"))
        if contract is not None and (
            type(contract) is not int or contract != 1
            or any(type(raw.get(field)) is not int for field in
                   ("candidate_tip_count", "branch_cluster_count", "skeleton_component_count"))
            or raw.get("geometry_pair_eligible") is not True
            or raw.get("candidate_tip_count") != 2
            or raw.get("branch_cluster_count") != 0
            or raw.get("skeleton_component_count") != 1
            or raw.get("candidate_tips_xy") != raw_ends
            or raw.get("endpoint_status") != "unbranched_two_tip_visible_segment"
        ):
            item["state"] = "unverified_visible_segment_geometry"
            evidence_records.append(item)
            continue
        item["geometry_evidence_verified"] = contract == 1
        if not isinstance(raw_ends, list) or len(raw_ends) != 2:
            item["state"] = "ambiguous_visible_segment_end_count"
            item["visible_end_count"] = len(raw_ends) if isinstance(raw_ends, list) else None
            evidence_records.append(item)
            continue

        ends = [_endpoint(point, f"{record_id} visible_ends_xy") for point in raw_ends]
        matches = [_matches(point, terminals, padding) for point in ends]
        item["visible_ends_xy"] = [[point[0], point[1]] for point in ends]
        item["terminal_matches"] = matches
        if any(len(hit) == 0 for hit in matches):
            item["state"] = "endpoint_unassigned"
        elif any(len(hit) > 1 for hit in matches):
            item["state"] = "endpoint_matches_multiple_terminals"
        elif matches[0][0] == matches[1][0]:
            item["state"] = "same_terminal_both_ends"
        else:
            edge_key = tuple(sorted((matches[0][0], matches[1][0])))
            edges[edge_key].append((record_id, score))
            item["state"] = "candidate_connection"
            item["candidate_edge"] = list(edge_key)
        evidence_records.append(item)

    raw_connections: list[dict[str, Any]] = []
    for edge_index, (edge_key, edge_evidence) in enumerate(sorted(edges.items()), start=1):
        raw_connections.append(
            {
                "id": f"sam_edge_{edge_index:03d}",
                "from": edge_key[0],
                "to": edge_key[1],
                "confidence": max(score for _, score in edge_evidence),
                "evidence_ids": [record_id for record_id, _ in edge_evidence],
            }
        )

    graph = ConnectionGraph.from_dict(
        {
            "schema_version": 1,
            "graph_id": graph_id,
            "scene_type": scene_type,
            "nodes": terminals,
            "connections": raw_connections,
        }
    )
    candidate_record_count = sum(
        item["state"] == "candidate_connection" for item in evidence_records
    )
    report = {
        "schema_version": 1,
        "adapter": "sam3_visible_segment_endpoints_to_declared_terminal_graph",
        "source_dir": endpoint_report.get("source_dir"),
        "endpoint_padding_px": padding,
        "observed_graph_id": graph.graph_id,
        "automatic_fault_verdict": False,
        "manual_confirmation_required": True,
        "summary": {
            "endpoint_record_count": len(evidence_records),
            "candidate_record_count": candidate_record_count,
            "emitted_connection_count": len(raw_connections),
            "unresolved_record_count": len(evidence_records) - candidate_record_count,
            "merged_duplicate_evidence_count": candidate_record_count - len(raw_connections),
            "legacy_geometry_unverified_record_count": sum(
                "geometry_contract_version" not in raw
                and "geometry_contract_version" not in endpoint_report for raw in raw_records
            ),
        },
        "evidence_records": evidence_records,
        "not_claimed": [
            "physical cable identity",
            "electrical continuity",
            "hidden path reconstruction",
            "automatic fault verdict",
            "verified geometry for legacy records without geometry metadata",
        ],
    }
    return graph, report
