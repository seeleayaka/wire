"""Regression harness for full-cabinet DINO candidate noise.

This replays the current cable-cabinet recipe without a GUI.  It locks down the
user-reported failure: wrong4 must retain its real central wire candidate while
edge/rail/label noise may not expand into a collection of unrelated boxes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Torch must load before the perspective module imports PyQt on this Windows host.
from tiled_dino_review import review_components
from assembly_auto_review_dino_v2 import dino_fused_regions
import assembly_auto_review_robust as robust
import assembly_auto_review_robust_v3 as perspective


PROJECT = Path(__file__).resolve().parents[1]
REGRESSION_SUITE = PROJECT / "config" / "regression_suite.json"
CABINET_RECIPE = PROJECT / "config" / "cabinet_dino_review_recipe.json"
MEMORY2_LARGE_ROI = [[367 / 1280, 894 / 1706, 978 / 1280, 1473 / 1706]]
MEMORY2_GROUPED_ROI = [[404 / 1280, 937 / 1706, 920 / 1280, 1427 / 1706]]
MEMORY2_LARGE_TARGETS = {
    "memory2-large-wrong-13": [[404, 1026, 434, 1414]],
    "memory2-large-wrong-14": [[404, 1160, 445, 1285], [404, 1383, 465, 1443], [508, 1146, 538, 1173]],
}
# These are the two physical repeated-structure groups in wrong-14.  This is
# deliberately stricter than asserting that a few residual fragments exist:
# each group needs its own final review rectangle, while the UI stays concise.
MEMORY2_GROUPED_TARGETS = {
    "left_memory_bank": [404, 957, 574, 1427],
    "right_memory_bank": [736, 965, 920, 1350],
}


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read {path}")
    return image


def load_cabinet_recipe() -> tuple[Path, list[list[float]], list[str]]:
    """Read the same explicit cabinet recipe used by the DINO operator UI."""
    recipe = json.loads(CABINET_RECIPE.read_text(encoding="utf-8"))
    reference = Path(str(recipe["reference_image"]))
    rois = recipe.get("check_rois", [])
    smoke_images = [str(name) for name in recipe.get("synthetic_smoke_images", [])]
    if not reference.is_file():
        raise FileNotFoundError(f"cabinet reference image is missing: {reference}")
    if not rois:
        raise ValueError(f"cabinet recipe has no check_rois: {CABINET_RECIPE}")
    return reference, rois, smoke_images


def pixels(image: np.ndarray, roi: list[float]) -> tuple[int, int, int, int]:
    height, width = image.shape[:2]
    left, top, right, bottom = roi
    return round(left * width), round(top * height), round(right * width), round(bottom * height)


def overlaps(first: list[int], second: list[int]) -> bool:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    return right > left and bottom > top


def coverage_of(target: list[int], candidate: list[int]) -> float:
    left = max(target[0], candidate[0])
    top = max(target[1], candidate[1])
    right = min(target[2], candidate[2])
    bottom = min(target[3], candidate[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    area = max(1, (target[2] - target[0]) * (target[3] - target[1]))
    return intersection / float(area)


def candidates_for(
    reference: Path, inspection: Path, rois: list[list[float]], *, real_pipeline: bool = False
) -> dict[str, Any]:
    ref, test = read_image(reference), read_image(inspection)
    aligned, alignment = perspective.automatic_homography(ref, test)
    result: dict[str, Any] = {"reference": str(reference), "inspection": str(inspection), "alignment": alignment, "candidates": []}
    if aligned is None:
        return result
    if real_pipeline:
        _overlay, _heat, candidates = dino_fused_regions(ref, aligned, rois)
        result["candidates"] = candidates
        result["local_alignment"] = robust.LAST_DIAGNOSTICS
        result["pipeline_mode"] = "real_dino_pipeline"
        return result
    valid_mask = getattr(perspective.auto, "LAST_WARP_VALID_MASK", None)
    for roi_index, roi in enumerate(rois, 1):
        left, top, right, bottom = pixels(ref, roi)
        roi_area_ratio = ((right - left) * (bottom - top)) / float(ref.shape[0] * ref.shape[1])
        valid_part = valid_mask[top:bottom, left:right] if valid_mask is not None else None
        _, metadata, candidates = review_components(
            ref[top:bottom, left:right],
            aligned[top:bottom, left:right],
            valid_part,
            reference_size=(ref.shape[1], ref.shape[0]),
            roi_area_ratio=roi_area_ratio,
            require_whole_cross_evidence=roi_area_ratio >= 0.50,
        )
        for candidate in candidates:
            result["candidates"].append({
                "roi": roi_index,
                "left": candidate["left"] + left,
                "top": candidate["top"] + top,
                "right": candidate["right"] + left,
                "bottom": candidate["bottom"] + top,
                "area": candidate["area"],
                "difference_score": candidate["difference_score"],
                "source_tiles": candidate.get("source_tiles", ["whole_roi"]),
                "evidence_summary": candidate.get("evidence_summary", {}),
            })
        result.setdefault("dino", []).append(metadata)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--real-pipeline", action="store_true", help="Run through dino_fused_regions, including local ECC")
    parser.add_argument("--cabinet-only", action="store_true", help="Skip unrelated memory2 regressions.")
    args = parser.parse_args()
    cases: list[dict[str, Any]] = []
    cable_reference, cable_review_rois, smoke_images = load_cabinet_recipe()
    normal = candidates_for(cable_reference, cable_reference, cable_review_rois, real_pipeline=args.real_pipeline)
    cases.append({
        "id": "cable-right-self",
        **normal,
        "expected_candidate_count": 0,
        "passed": len(normal["candidates"]) == 0,
    })
    for name in smoke_images:
        result = candidates_for(cable_reference, cable_reference.parent / name, cable_review_rois, real_pipeline=args.real_pipeline)
        item: dict[str, Any] = {"id": f"cable-{name}", **result}
        item["passed"] = len(item["candidates"]) >= 1
        if name == "wrong4.png":
            expected_wire = [500, 160, 640, 310]
            item["expected_wire_region"] = expected_wire
            item["target_retained"] = any(overlaps([candidate[key] for key in ("left", "top", "right", "bottom")], expected_wire) for candidate in item["candidates"])
            item["noise_budget_ok"] = len(item["candidates"]) <= 3
            item["passed"] = bool(item["passed"] and item["target_retained"] and item["noise_budget_ok"])
        if name == "wrong5.png":
            item["noise_budget_ok"] = len(item["candidates"]) <= 1
            item["passed"] = bool(item["passed"] and item["noise_budget_ok"])
        cases.append(item)

    if args.cabinet_only:
        report = {"total": len(cases), "passed": sum(bool(case["passed"]) for case in cases), "failed": sum(not bool(case["passed"]) for case in cases), "cases": cases}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(args.output), "total": report["total"], "passed": report["passed"], "failed": report["failed"]}, ensure_ascii=False))
        if report["failed"]:
            raise SystemExit(1)
        return

    suite = json.loads(REGRESSION_SUITE.read_text(encoding="utf-8"))
    for case in suite["cases"]:
        if not case["id"].startswith("memory2-"):
            continue
        result = candidates_for(Path(case["reference"]), Path(case["inspection"]), case["rois"], real_pipeline=args.real_pipeline)
        cases.append({"id": case["id"], **result, "passed": len(result["candidates"]) >= 1})

    memory2 = {case["id"]: case for case in suite["cases"] if case["id"] in {"memory2-wrong-13", "memory2-wrong-14"}}
    for source_id, case in memory2.items():
        large_id = source_id.replace("memory2-", "memory2-large-")
        result = candidates_for(Path(case["reference"]), Path(case["inspection"]), MEMORY2_LARGE_ROI, real_pipeline=args.real_pipeline)
        targets = MEMORY2_LARGE_TARGETS[large_id]
        target_retained = any(
            overlaps([candidate[key] for key in ("left", "top", "right", "bottom")], target)
            for candidate in result["candidates"]
            for target in targets
        )
        cases.append({
            "id": large_id,
            **result,
            "expected_regions": targets,
            "target_retained": target_retained,
            "passed": bool(result["candidates"] and target_retained),
        })

    wrong14 = memory2["memory2-wrong-14"]
    grouped = candidates_for(Path(wrong14["reference"]), Path(wrong14["inspection"]), MEMORY2_GROUPED_ROI, real_pipeline=args.real_pipeline)
    grouped_candidates = [
        [candidate[key] for key in ("left", "top", "right", "bottom")]
        for candidate in grouped["candidates"]
    ]
    group_coverage = {
        name: max((coverage_of(target, candidate) for candidate in grouped_candidates), default=0.0)
        for name, target in MEMORY2_GROUPED_TARGETS.items()
    }
    # A group-level frame may leave a small visual margin around the hardware,
    # but a residual sliver cannot count as detection of the physical group.
    grouped_hit = {name: coverage >= 0.55 for name, coverage in group_coverage.items()}
    cases.append({
        "id": "memory2-grouped-wrong-14",
        **grouped,
        "expected_groups": MEMORY2_GROUPED_TARGETS,
        "group_coverage": group_coverage,
        "grouped_hit": grouped_hit,
        "candidate_budget_ok": len(grouped_candidates) == 2,
        "passed": bool(all(grouped_hit.values()) and len(grouped_candidates) == 2),
    })

    report = {"total": len(cases), "passed": sum(bool(case["passed"]) for case in cases), "failed": sum(not case["passed"] for case in cases), "cases": cases}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "total": report["total"], "passed": report["passed"], "failed": report["failed"]}, ensure_ascii=False))
    if report["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
