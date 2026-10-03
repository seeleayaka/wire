"""Small, explicit connection-graph model for controlled inspection scenes.

The first version intentionally compares declared terminal-to-terminal edges.
It does not infer electrical continuity from pixels and does not turn visual
candidates into a fault verdict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


class TopologyValidationError(ValueError):
    """Raised when a topology document is ambiguous or internally invalid."""


def _non_empty_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TopologyValidationError(f"{field} must be a non-empty string")
    return value.strip()


def _edge_key(first: str, second: str) -> tuple[str, str]:
    if first == second:
        raise TopologyValidationError("a connection cannot join a node to itself")
    return tuple(sorted((first, second)))


@dataclass(frozen=True)
class Connection:
    first: str
    second: str
    connection_id: str
    confidence: float | None = None
    evidence_ids: tuple[str, ...] = ()

    @property
    def key(self) -> tuple[str, str]:
        return _edge_key(self.first, self.second)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.connection_id,
            "from": self.first,
            "to": self.second,
        }
        if self.confidence is not None:
            result["confidence"] = self.confidence
        if self.evidence_ids:
            result["evidence_ids"] = list(self.evidence_ids)
        return result


@dataclass(frozen=True)
class ConnectionGraph:
    graph_id: str
    scene_type: str
    nodes: tuple[dict[str, Any], ...]
    connections: tuple[Connection, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ConnectionGraph":
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            raise TopologyValidationError("topology schema_version must be 1")
        graph_id = _non_empty_text(value.get("graph_id"), "graph_id")
        scene_type = _non_empty_text(value.get("scene_type"), "scene_type")
        raw_nodes = value.get("nodes")
        raw_connections = value.get("connections")
        if not isinstance(raw_nodes, list) or not isinstance(raw_connections, list):
            raise TopologyValidationError("nodes and connections must be lists")

        nodes: list[dict[str, Any]] = []
        node_ids: set[str] = set()
        for index, raw_node in enumerate(raw_nodes, start=1):
            if not isinstance(raw_node, dict):
                raise TopologyValidationError(f"node {index} must be an object")
            node_id = _non_empty_text(raw_node.get("id"), f"node {index} id")
            if node_id in node_ids:
                raise TopologyValidationError(f"duplicate node id: {node_id}")
            node_ids.add(node_id)
            node = dict(raw_node)
            node["id"] = node_id
            nodes.append(node)

        connections: list[Connection] = []
        edge_keys: set[tuple[str, str]] = set()
        for index, raw_connection in enumerate(raw_connections, start=1):
            if not isinstance(raw_connection, dict):
                raise TopologyValidationError(f"connection {index} must be an object")
            first = _non_empty_text(raw_connection.get("from"), f"connection {index} from")
            second = _non_empty_text(raw_connection.get("to"), f"connection {index} to")
            if first not in node_ids or second not in node_ids:
                raise TopologyValidationError(
                    f"connection {index} references an undeclared node: {first}, {second}"
                )
            key = _edge_key(first, second)
            if key in edge_keys:
                raise TopologyValidationError(f"duplicate connection: {key[0]} <-> {key[1]}")
            edge_keys.add(key)
            confidence = raw_connection.get("confidence")
            if confidence is not None:
                if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
                    raise TopologyValidationError(f"connection {index} confidence must be numeric")
                confidence = float(confidence)
                if not 0.0 <= confidence <= 1.0:
                    raise TopologyValidationError(f"connection {index} confidence must be within 0..1")
            raw_evidence = raw_connection.get("evidence_ids", [])
            if not isinstance(raw_evidence, list) or not all(
                isinstance(item, str) and item.strip() for item in raw_evidence
            ):
                raise TopologyValidationError(f"connection {index} evidence_ids must be strings")
            connections.append(
                Connection(
                    first=first,
                    second=second,
                    connection_id=_non_empty_text(
                        raw_connection.get("id", f"edge_{index:03d}"),
                        f"connection {index} id",
                    ),
                    confidence=confidence,
                    evidence_ids=tuple(item.strip() for item in raw_evidence),
                )
            )
        return cls(graph_id, scene_type, tuple(nodes), tuple(connections))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "graph_id": self.graph_id,
            "scene_type": self.scene_type,
            "nodes": [dict(node) for node in self.nodes],
            "connections": [connection.to_dict() for connection in self.connections],
        }


def _edge_records(edges: Iterable[Connection]) -> list[dict[str, Any]]:
    return [edge.to_dict() for edge in sorted(edges, key=lambda item: item.key)]


def compare_topologies(
    expected: ConnectionGraph,
    observed: ConnectionGraph,
    *,
    minimum_confidence: float = 0.75,
) -> dict[str, Any]:
    """Compare explicit undirected connections while preserving uncertainty."""
    if expected.scene_type != observed.scene_type:
        raise TopologyValidationError("expected and observed topology scene_type values differ")
    if not 0.0 <= minimum_confidence <= 1.0:
        raise TopologyValidationError("minimum_confidence must be within 0..1")

    expected_by_key = {connection.key: connection for connection in expected.connections}
    confident_observed: dict[tuple[str, str], Connection] = {}
    uncertain: list[Connection] = []
    for connection in observed.connections:
        if connection.confidence is not None and connection.confidence < minimum_confidence:
            uncertain.append(connection)
        else:
            confident_observed[connection.key] = connection

    missing = [edge for key, edge in expected_by_key.items() if key not in confident_observed]
    unexpected = [edge for key, edge in confident_observed.items() if key not in expected_by_key]
    expected_keys = set(expected_by_key)
    observed_keys = set(confident_observed)
    edge_edit_count = len(expected_keys.symmetric_difference(observed_keys))
    edge_union_count = len(expected_keys.union(observed_keys))
    normalized_edge_distance = (
        edge_edit_count / edge_union_count if edge_union_count else 0.0
    )
    if uncertain:
        decision = "insufficient_evidence"
    elif missing or unexpected:
        decision = "mismatch_manual_confirmation_required"
    else:
        decision = "match"
    return {
        "schema_version": 1,
        "decision": decision,
        "automatic_fault_verdict": False,
        "minimum_confidence": minimum_confidence,
        "expected_graph_id": expected.graph_id,
        "observed_graph_id": observed.graph_id,
        "missing_connections": _edge_records(missing),
        "unexpected_connections": _edge_records(unexpected),
        "uncertain_connections": _edge_records(uncertain),
        "summary": {
            "expected_connection_count": len(expected.connections),
            "confident_observed_connection_count": len(confident_observed),
            "missing_count": len(missing),
            "unexpected_count": len(unexpected),
            "uncertain_count": len(uncertain),
        },
        "topology_distance": {
            "metric": "normalized_symmetric_edge_difference",
            "inspiration": "Image2Net Netlist Edit Distance (NED)",
            "is_image2net_ned": False,
            "edge_edit_count": edge_edit_count,
            "edge_union_count": edge_union_count,
            "normalized_distance": normalized_edge_distance,
            "similarity": 1.0 - normalized_edge_distance,
            "uncertain_connections_excluded": len(uncertain),
            "interpretation": "0 means identical declared edges; 1 means no declared edge overlaps",
        },
    }
