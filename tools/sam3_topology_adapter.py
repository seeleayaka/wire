"""Convert a SAM3 endpoint report plus declared terminal ROIs to a graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inspection_agent import (  # noqa: E402
    ConnectionGraph,
    compare_topologies,
    connection_graph_from_sam_endpoints,
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Conservatively map SAM3 visible-segment endpoints to declared terminal ROIs"
    )
    parser.add_argument("--endpoint-report", type=Path, required=True)
    parser.add_argument("--terminal-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected", type=Path)
    parser.add_argument("--endpoint-padding-px", type=float, default=0.0)
    parser.add_argument("--minimum-confidence", type=float, default=0.75)
    args = parser.parse_args()

    observed, adapter_report = connection_graph_from_sam_endpoints(
        _load(args.endpoint_report),
        _load(args.terminal_map),
        endpoint_padding_px=args.endpoint_padding_px,
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "observed_graph": observed.to_dict(),
        "adapter_report": adapter_report,
    }
    if args.expected is not None:
        expected = ConnectionGraph.from_dict(_load(args.expected))
        result["comparison"] = compare_topologies(
            expected,
            observed,
            minimum_confidence=args.minimum_confidence,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "summary": adapter_report["summary"],
                "decision": result.get("comparison", {}).get("decision"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
