"""Exercise the lightweight topology comparator on all Image2Net golden JSON.

Exact copies, one deleted port connection, and one rewired port connection are
evaluated separately.  The perturbations are controlled contract tests, not
predictions from circuit images.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import statistics
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inspection_agent import compare_topologies, connection_graph_from_image2net  # noqa: E402


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _port_locations(value: dict[str, Any]) -> list[tuple[int, str, str]]:
    result: list[tuple[int, str, str]] = []
    for component_index, component in enumerate(value["ckt_netlist"]):
        for port_name, net_name in component["port_connection"].items():
            result.append((component_index, port_name, net_name))
    return result


def _delete_one(value: dict[str, Any]) -> dict[str, Any]:
    changed = deepcopy(value)
    component_index, port_name, _ = _port_locations(changed)[0]
    del changed["ckt_netlist"][component_index]["port_connection"][port_name]
    return changed


def _rewire_one(value: dict[str, Any]) -> dict[str, Any] | None:
    changed = deepcopy(value)
    ports = _port_locations(changed)
    all_nets = sorted({net_name for _, _, net_name in ports})
    for component_index, port_name, original_net in ports:
        alternative = next((net for net in all_nets if net != original_net), None)
        if alternative is not None:
            changed["ckt_netlist"][component_index]["port_connection"][port_name] = alternative
            return changed
    return None


def _distance(expected_value: dict[str, Any], observed_value: dict[str, Any], stem: str) -> float:
    expected = connection_graph_from_image2net(expected_value, graph_id=f"{stem}:expected")
    observed = connection_graph_from_image2net(observed_value, graph_id=f"{stem}:observed")
    return float(compare_topologies(expected, observed)["topology_distance"]["normalized_distance"])


def evaluate(golden_dir: Path) -> dict[str, Any]:
    per_case: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for path in sorted(golden_dir.glob("*.json")):
        try:
            value = _load(path)
            exact = _distance(value, value, path.stem)
            deleted = _distance(value, _delete_one(value), path.stem)
            rewired_value = _rewire_one(value)
            rewired = None if rewired_value is None else _distance(value, rewired_value, path.stem)
            per_case.append(
                {
                    "case_id": path.stem,
                    "connection_count": len(_port_locations(value)),
                    "exact_distance": exact,
                    "one_deleted_distance": deleted,
                    "one_rewired_distance": rewired,
                }
            )
        except Exception as exc:  # report every malformed public case without hiding the rest
            failures.append({"case_id": path.stem, "error": f"{type(exc).__name__}: {exc}"})

    deletion_distances = [item["one_deleted_distance"] for item in per_case]
    rewired_distances = [item["one_rewired_distance"] for item in per_case if item["one_rewired_distance"] is not None]
    exact_zero = sum(item["exact_distance"] == 0.0 for item in per_case)
    deletion_detected = sum(distance > 0.0 for distance in deletion_distances)
    rewiring_detected = sum(distance > 0.0 for distance in rewired_distances)
    return {
        "schema_version": 1,
        "benchmark": "image2net_order_preserving_connection_graph",
        "status": "ok" if not failures and exact_zero == len(per_case) and deletion_detected == len(deletion_distances) and rewiring_detected == len(rewired_distances) else "failed",
        "summary": {
            "case_count": len(per_case),
            "failure_count": len(failures),
            "exact_zero_distance_count": exact_zero,
            "one_deleted_detected_count": deletion_detected,
            "one_rewired_case_count": len(rewired_distances),
            "one_rewired_detected_count": rewiring_detected,
            "mean_one_deleted_distance": statistics.fmean(deletion_distances) if deletion_distances else None,
            "mean_one_rewired_distance": statistics.fmean(rewired_distances) if rewired_distances else None,
        },
        "failures": failures,
        "per_case": per_case,
        "evidence_boundary": "controlled JSON perturbations; not diagram recognition and not Image2Net GED/NED",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the lightweight topology metric on Image2Net golden JSON")
    parser.add_argument("--golden-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate(args.golden_dir)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "summary": report["summary"]}, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
