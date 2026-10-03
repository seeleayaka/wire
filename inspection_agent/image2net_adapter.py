"""Order-preserving adapter for the public Image2Net validation netlists.

This adapter is deliberately lightweight.  It maps every component port to a
declared net so the existing connection-graph comparator can be exercised on
all public golden JSON files.  It does not implement Image2Net's heterogeneous
graph isomorphism or claim equivalence to its GED/NED benchmark.
"""

from __future__ import annotations

from typing import Any

from .topology import ConnectionGraph, TopologyValidationError


class Image2NetAdapterError(TopologyValidationError):
    """Raised when an Image2Net netlist cannot be mapped unambiguously."""


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Image2NetAdapterError(f"{field} must be a non-empty string")
    return value.strip()


def connection_graph_from_image2net(
    value: dict[str, Any],
    *,
    graph_id: str,
) -> ConnectionGraph:
    """Map ordered component ports to a bipartite port/net connection graph."""
    if not isinstance(value, dict):
        raise Image2NetAdapterError("Image2Net document must be an object")
    raw_components = value.get("ckt_netlist")
    if not isinstance(raw_components, list) or not raw_components:
        raise Image2NetAdapterError("ckt_netlist must be a non-empty list")

    port_records: list[tuple[str, str, str, int]] = []
    net_names: set[str] = set()
    nodes: list[dict[str, Any]] = []
    for component_index, raw_component in enumerate(raw_components):
        if not isinstance(raw_component, dict):
            raise Image2NetAdapterError(f"component {component_index} must be an object")
        component_type = _text(
            raw_component.get("component_type"),
            f"component {component_index} component_type",
        )
        raw_ports = raw_component.get("port_connection")
        if not isinstance(raw_ports, dict) or not raw_ports:
            raise Image2NetAdapterError(
                f"component {component_index} port_connection must be a non-empty object"
            )
        for port_name_raw, net_name_raw in raw_ports.items():
            port_name = _text(port_name_raw, f"component {component_index} port name")
            net_name = _text(
                net_name_raw,
                f"component {component_index} port {port_name} net",
            )
            port_id = f"port:{component_index:04d}:{component_type}:{port_name}"
            nodes.append(
                {
                    "id": port_id,
                    "kind": "component_port",
                    "component_index": component_index,
                    "component_type": component_type,
                    "port_name": port_name,
                }
            )
            port_records.append((port_id, net_name, port_name, component_index))
            net_names.add(net_name)

    for net_name in sorted(net_names):
        nodes.append({"id": f"net:{net_name}", "kind": "net", "net_name": net_name})

    connections = [
        {
            "id": f"link:{component_index:04d}:{port_name}",
            "from": port_id,
            "to": f"net:{net_name}",
        }
        for port_id, net_name, port_name, component_index in port_records
    ]
    return ConnectionGraph.from_dict(
        {
            "schema_version": 1,
            "graph_id": graph_id,
            "scene_type": "image2net_ordered_netlist",
            "nodes": nodes,
            "connections": connections,
        }
    )
