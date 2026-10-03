"""Associate visible-segment endpoint markers with existing SAM3 difference candidates.

This is an evidence-only rule layer. It accepts endpoints only from the exact
SAM3 directories named in an accepted mainline report, so reference-coordinate
candidate boxes are never correlated with pre-alignment inspection masks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


class EndpointRuleError(RuntimeError):
    """Input-contract failure for the evidence-only endpoint rule layer."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mainline-report", type=Path, required=True)
    parser.add_argument("--reference-endpoints", type=Path, required=True)
    parser.add_argument("--inspection-endpoints", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint-padding-px", type=int, default=16)
    return parser.parse_args()


def load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise EndpointRuleError(f"{label} is unavailable")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise EndpointRuleError(f"{label} is invalid JSON") from error
    if not isinstance(data, dict):
        raise EndpointRuleError(f"{label} must be a JSON object")
    return data


def _same_path(left: Any, right: Any) -> bool:
    return isinstance(left, str) and Path(left).resolve() == Path(str(right)).resolve()


def validate_mainline_contract(
    mainline: dict[str, Any],
    reference_endpoints: dict[str, Any],
    inspection_endpoints: dict[str, Any],
) -> None:
    alignment = mainline.get("alignment_quality")
    fusion = mainline.get("sam3_fusion")
    if not isinstance(alignment, dict) or alignment.get("reliable") is not True:
        raise EndpointRuleError("mainline alignment was not accepted")
    if not isinstance(fusion, dict) or fusion.get("status") != "ok":
        raise EndpointRuleError("mainline SAM3 fusion was not successful")
    reference = fusion.get("reference_sam3")
    inspection = fusion.get("inspection_sam3")
    if not isinstance(reference, dict) or not isinstance(inspection, dict):
        raise EndpointRuleError("mainline report lacks SAM3 source directories")
    if not _same_path(reference_endpoints.get("source_dir"), reference.get("output_dir")):
        raise EndpointRuleError("reference endpoint report is not from this mainline SAM3 reference output")
    if not _same_path(inspection_endpoints.get("source_dir"), inspection.get("output_dir")):
        raise EndpointRuleError("inspection endpoint report is not from this mainline aligned SAM3 output")


def _integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise EndpointRuleError(f"{label} must be an integer")
    return value


def endpoint_records(report: dict[str, Any], label: str) -> list[dict[str, Any]]:
    raw_records = report.get("records")
    if not isinstance(raw_records, list):
        raise EndpointRuleError(f"{label} endpoint report has no records")
    normalized: list[dict[str, Any]] = []
    for record in raw_records:
        if not isinstance(record, dict) or not isinstance(record.get("record_id"), str):
            raise EndpointRuleError(f"{label} endpoint record is invalid")
        points = record.get("visible_ends_xy")
        if not isinstance(points, list):
            raise EndpointRuleError(f"{label} endpoint record has no visible ends")
        normalized_points: list[list[int]] = []
        for point in points:
            if not isinstance(point, list) or len(point) != 2:
                raise EndpointRuleError(f"{label} endpoint coordinate is invalid")
            normalized_points.append([_integer(point[0], "endpoint x"), _integer(point[1], "endpoint y")])
        normalized.append(
            {
                "record_id": record["record_id"],
                "visible_ends_xy": normalized_points,
                "component_pixels": record.get("component_pixels"),
                "endpoint_status": record.get("endpoint_status", "unclassified_visible_segment_end"),
            }
        )
    return normalized


def candidate_records(report: dict[str, Any]) -> list[dict[str, Any]]:
    raw_candidates = report.get("review_regions", report.get("candidates"))
    if not isinstance(raw_candidates, list):
        raise EndpointRuleError("candidate report has no review_regions or candidates array")
    normalized: list[dict[str, Any]] = []
    for index, candidate in enumerate(raw_candidates, 1):
        if not isinstance(candidate, dict):
            raise EndpointRuleError("candidate is invalid")
        bbox = candidate.get("bbox_xyxy")
        if isinstance(bbox, list) and len(bbox) == 4:
            left, top, right, bottom = (_integer(value, "candidate bbox coordinate") for value in bbox)
        else:
            left = _integer(candidate.get("left"), "candidate left")
            top = _integer(candidate.get("top"), "candidate top")
            right = _integer(candidate.get("right"), "candidate right")
            bottom = _integer(candidate.get("bottom"), "candidate bottom")
        if right <= left or bottom <= top:
            raise EndpointRuleError("candidate box is empty")
        normalized.append(
            {
                "candidate_id": candidate["id"] if isinstance(candidate.get("id"), str) and candidate["id"] else f"candidate_{index:03d}",
                "bbox_xyxy": [left, top, right, bottom],
                "direction": candidate.get("direction"),
                "local_evidence": {
                    key: candidate[key]
                    for key in ("difference_score", "area", "component_pixels", "sam_added_pixels", "sam_missing_pixels", "sam_support_pixels")
                    if isinstance(candidate.get(key), (int, float)) and not isinstance(candidate.get(key), bool)
                },
            }
        )
    return normalized


def _endpoint_hits(records: list[dict[str, Any]], bbox: list[int], padding: int) -> list[dict[str, Any]]:
    left, top, right, bottom = bbox
    hits: list[dict[str, Any]] = []
    for record in records:
        point_hits = []
        for index, (x, y) in enumerate(record["visible_ends_xy"], 1):
            if left - padding <= x <= right + padding and top - padding <= y <= bottom + padding:
                point_hits.append({"endpoint_index": index, "xy": [x, y]})
        if point_hits:
            hits.append(
                {
                    "record_id": record["record_id"],
                    "endpoint_hits": point_hits,
                    "endpoint_status": record["endpoint_status"],
                }
            )
    return hits


def build_endpoint_evidence(
    reference_records: list[dict[str, Any]],
    inspection_records: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    padding: int,
) -> dict[str, Any]:
    """Apply only a spatial endpoint-near-candidate rule, without segment joining."""
    if padding < 0:
        raise EndpointRuleError("endpoint padding must be non-negative")
    evidence_candidates = []
    for candidate in candidates:
        reference_hits = _endpoint_hits(reference_records, candidate["bbox_xyxy"], padding)
        inspection_hits = _endpoint_hits(inspection_records, candidate["bbox_xyxy"], padding)
        state = "visible_segment_endpoints_near_candidate_manual_review" if reference_hits or inspection_hits else "no_visible_segment_endpoint_near_candidate_manual_review"
        evidence_candidates.append(
            {
                **candidate,
                "reference_visible_segment_endpoints": reference_hits,
                "inspection_visible_segment_endpoints": inspection_hits,
                "endpoint_rule_state": state,
            }
        )
    return {
        "candidate_count": len(evidence_candidates),
        "candidates_with_reference_endpoint_evidence": sum(bool(item["reference_visible_segment_endpoints"]) for item in evidence_candidates),
        "candidates_with_inspection_endpoint_evidence": sum(bool(item["inspection_visible_segment_endpoints"]) for item in evidence_candidates),
        "candidates": evidence_candidates,
    }


def main() -> int:
    args = parse_args()
    mainline = load_json(args.mainline_report, "mainline report")
    reference_report = load_json(args.reference_endpoints, "reference endpoint report")
    inspection_report = load_json(args.inspection_endpoints, "inspection endpoint report")
    candidates_report = load_json(args.candidate_report, "candidate report")
    validate_mainline_contract(mainline, reference_report, inspection_report)
    evidence = build_endpoint_evidence(
        endpoint_records(reference_report, "reference"),
        endpoint_records(inspection_report, "inspection"),
        candidate_records(candidates_report),
        args.endpoint_padding_px,
    )
    result = {
        "purpose": "Endpoint-near-candidate evidence from independent SAM3 visible segments after accepted alignment; manual review only.",
        "coordinate_contract": "Endpoint reports must originate from the exact reference and aligned-inspection SAM3 directories in the accepted mainline report.",
        "rules": {
            "endpoint_padding_px": args.endpoint_padding_px,
            "endpoint_rule": "An endpoint marker is recorded only when it lies inside or near an existing local difference candidate box.",
            "no_mask_joining": True,
        },
        "not_claimed": [
            "physical_cable_identity",
            "cross_mask_path_reconstruction",
            "terminal_assignment",
            "electrical_continuity",
            "fault_type",
            "automatic_acceptance_or_rejection",
        ],
        "manual_confirmation_required": True,
        "endpoint_evidence": evidence,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: evidence[key] for key in evidence if key != "candidates"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
