"""Headless regression runner for the reference-vs-inspection review pipeline."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

# PyTorch must be loaded before the perspective module imports PyQt.  On this
# Windows environment the reverse order can make torch's c10.dll fail to
# initialize (WinError 1114).
import cv2
import numpy as np

from tiled_dino_review import review_components
from assembly_auto_review_dino_v2 import dino_fused_regions
import assembly_auto_review_robust as robust
import assembly_auto_review_robust_v3 as perspective


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read {path}")
    return image


def pixels(image: np.ndarray, roi: list[float]) -> tuple[int, int, int, int]:
    height, width = image.shape[:2]
    left, top, right, bottom = roi
    return round(left * width), round(top * height), round(right * width), round(bottom * height)


def evaluate_case(case: dict[str, Any], *, real_pipeline: bool = False) -> dict[str, Any]:
    reference_path, inspection_path = Path(case["reference"]), Path(case["inspection"])
    result: dict[str, Any] = {"id": case["id"], "expected_decision": case["expected_decision"]}
    if not reference_path.is_file() or not inspection_path.is_file():
        result.update(decision="fixture_missing", passed=False, missing=[str(path) for path in (reference_path, inspection_path) if not path.is_file()])
        return result
    reference, inspection = read_image(reference_path), read_image(inspection_path)
    aligned, alignment = perspective.automatic_homography(reference, inspection)
    result["alignment"] = alignment
    if aligned is None:
        result["decision"] = "alignment_uncertain_manual_review"
        result["candidate_count"] = 0
    elif real_pipeline:
        rois = case.get("rois", [[0.0, 0.0, 1.0, 1.0]])
        _overlay, _heat, candidates = dino_fused_regions(reference, aligned, rois)
        result["candidate_count"] = len(candidates)
        result["decision"] = "possible_difference_manual_review" if candidates else "no_significant_difference"
        result["local_alignment"] = robust.LAST_DIAGNOSTICS
        result["pipeline_mode"] = "real_dino_pipeline"
    else:
        candidate_count = 0
        roi_reports = []
        valid_mask = getattr(perspective.auto, "LAST_WARP_VALID_MASK", None)
        for roi in case.get("rois", [[0.0, 0.0, 1.0, 1.0]]):
            left, top, right, bottom = pixels(reference, roi)
            roi_area_ratio = ((right - left) * (bottom - top)) / float(reference.shape[0] * reference.shape[1])
            valid_part = valid_mask[top:bottom, left:right] if valid_mask is not None else None
            score, metadata, candidates = review_components(
                reference[top:bottom, left:right],
                aligned[top:bottom, left:right],
                valid_part,
                reference_size=(reference.shape[1], reference.shape[0]),
                roi_area_ratio=roi_area_ratio,
                require_whole_cross_evidence=roi_area_ratio >= 0.50,
            )
            candidate_count += len(candidates)
            roi_reports.append({"roi": roi, "candidate_count": len(candidates), "dino": metadata})
        result["decision"] = "possible_difference_manual_review" if candidate_count else "no_significant_difference"
        result["candidate_count"] = candidate_count
        result["rois"] = roi_reports
    result["passed"] = result["decision"] == result["expected_decision"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=Path(__file__).resolve().parents[1] / "config" / "regression_suite.json")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "output" / "regression")
    parser.add_argument("--real-pipeline", action="store_true", help="Run through dino_fused_regions, including local ECC")
    args = parser.parse_args()
    suite = json.loads(args.suite.read_text(encoding="utf-8"))
    cases = [evaluate_case(case, real_pipeline=args.real_pipeline) for case in suite["cases"]]
    report = {"suite": str(args.suite), "created_at": datetime.now().isoformat(timespec="seconds"), "total": len(cases), "passed": sum(case["passed"] for case in cases), "failed": sum(not case["passed"] for case in cases), "cases": cases}
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / f"regression_{datetime.now():%Y%m%d_%H%M%S}.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "total": report["total"], "passed": report["passed"], "failed": report["failed"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
