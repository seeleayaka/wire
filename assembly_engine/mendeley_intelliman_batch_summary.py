"""Aggregate manual-only Mendeley-to-IntelliMan evidence reports.

The summary answers only whether existing IntelliMan topology geometry covered
Mendeley source-label boxes.  It deliberately preserves the original label
class IDs as opaque dataset IDs and never turns coverage into a fault score.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


class MendeleyIntelliManBatchSummaryError(ValueError):
    """Raised when an evidence report cannot safely enter a batch summary."""


def _report_object(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise MendeleyIntelliManBatchSummaryError(f"evidence report is unavailable: {source}")
    try:
        report = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise MendeleyIntelliManBatchSummaryError(f"invalid evidence report JSON: {source}") from error
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise MendeleyIntelliManBatchSummaryError(f"unsupported evidence report schema: {source}")
    if report.get("manual_evaluation_only") is not True:
        raise MendeleyIntelliManBatchSummaryError(f"report is not manual-evaluation-only: {source}")
    source_info, label_evidence = report.get("source"), report.get("label_evidence")
    if not isinstance(source_info, dict) or not isinstance(label_evidence, list):
        raise MendeleyIntelliManBatchSummaryError(f"report lacks source or label evidence: {source}")
    image_label = source_info.get("dataset_image_label", source_info.get("fault_family_from_filename"))
    if not isinstance(image_label, str) or not image_label:
        raise MendeleyIntelliManBatchSummaryError(f"report has no dataset image label: {source}")
    for index, evidence in enumerate(label_evidence, start=1):
        if not isinstance(evidence, dict):
            raise MendeleyIntelliManBatchSummaryError(f"report label {index} is invalid: {source}")
        if isinstance(evidence.get("source_class_id"), bool) or not isinstance(evidence.get("source_class_id"), int):
            raise MendeleyIntelliManBatchSummaryError(f"report label {index} has no integer source class ID: {source}")
        if not isinstance(evidence.get("topology_evidence_present"), bool):
            raise MendeleyIntelliManBatchSummaryError(f"report label {index} has no topology evidence flag: {source}")
    return report


def _empty_counts() -> dict[str, int]:
    return {
        "report_count": 0,
        "source_label_count": 0,
        "labels_with_topology_evidence": 0,
        "labels_without_topology_evidence": 0,
    }


def _add_label(counts: dict[str, int], evidence_present: bool) -> None:
    counts["source_label_count"] += 1
    if evidence_present:
        counts["labels_with_topology_evidence"] += 1
    else:
        counts["labels_without_topology_evidence"] += 1


def _with_coverage_ratio(counts: dict[str, int]) -> dict[str, int | float | None]:
    result: dict[str, int | float | None] = dict(counts)
    total = counts["source_label_count"]
    result["topology_evidence_coverage_ratio"] = round(counts["labels_with_topology_evidence"] / total, 6) if total else None
    return result


def summarize_mendeley_intelliman_evidence_reports(report_paths: Iterable[str | Path]) -> dict[str, Any]:
    """Return a transparent, non-predictive summary of evidence reports."""
    paths = [Path(path) for path in report_paths]
    if not paths:
        raise MendeleyIntelliManBatchSummaryError("at least one evidence report is required")

    overall = _empty_counts()
    by_image_label: dict[str, dict[str, int]] = defaultdict(_empty_counts)
    by_source_class: dict[str, dict[str, int]] = defaultdict(_empty_counts)
    source_reports: list[str] = []
    for path in paths:
        report = _report_object(path)
        source = report["source"]
        raw_image_label = source.get("dataset_image_label")
        if raw_image_label is None:
            raw_image_label = source["fault_family_from_filename"]
        image_label = str(raw_image_label)
        source_reports.append(str(path.resolve()))
        overall["report_count"] += 1
        by_image_label[image_label]["report_count"] += 1
        source_classes_in_report: set[str] = set()
        for evidence in report["label_evidence"]:
            evidence_present = bool(evidence["topology_evidence_present"])
            _add_label(overall, evidence_present)
            _add_label(by_image_label[image_label], evidence_present)
            source_class_id = str(evidence["source_class_id"])
            _add_label(by_source_class[source_class_id], evidence_present)
            source_classes_in_report.add(source_class_id)
        for source_class_id in source_classes_in_report:
            by_source_class[source_class_id]["report_count"] += 1

    return {
        "schema_version": 1,
        "purpose": "Aggregate Mendeley image-label and source-box topology coverage; manual evaluation only.",
        "manual_evaluation_only": True,
        "source_report_count": len(source_reports),
        "source_reports": source_reports,
        "overall": _with_coverage_ratio(overall),
        "by_dataset_image_label": {
            image_label: _with_coverage_ratio(by_image_label[image_label]) for image_label in sorted(by_image_label)
        },
        "by_opaque_source_class_id": {class_id: _with_coverage_ratio(by_source_class[class_id]) for class_id in sorted(by_source_class, key=int)},
        "not_claimed": [
            "model_fault_prediction",
            "automatic_fault_classification",
            "fault_class_accuracy",
            "dataset_image_label_prediction_accuracy",
            "cable_identity",
            "terminal_assignment",
            "electrical_continuity",
            "production_acceptance_or_rejection",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, action="append", required=True, help="manual-only evidence report; repeat for a batch")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize_mendeley_intelliman_evidence_reports(args.report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["overall"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
