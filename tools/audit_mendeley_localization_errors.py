"""Describe failure modes in a completed Mendeley localization report.

All labels are read after inference. This audit does not change candidates or
select thresholds; its purpose is to choose representative visual cases.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median


def area(box: dict | list) -> int:
    if isinstance(box, dict):
        left, top, right, bottom = (box[key] for key in ("left", "top", "right", "bottom"))
    else:
        left, top, right, bottom = box
    return max(0, right - left) * max(0, bottom - top)


def intersection(first: dict | list, second: dict | list) -> int:
    a = [first[key] for key in ("left", "top", "right", "bottom")] if isinstance(first, dict) else first
    b = [second[key] for key in ("left", "top", "right", "bottom")] if isinstance(second, dict) else second
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def summarize_case(case: dict) -> dict:
    candidates = case["candidates"]
    targets = case["target_boxes_aligned_xyxy"]
    hit_target_areas = []
    broad_candidates = []
    candidate_hits = 0
    for index, candidate in enumerate(candidates, 1):
        matched = [target for target in targets if intersection(candidate, target) > 0]
        candidate_hits += bool(matched)
        if matched:
            hit_target_areas.extend(area(target) for target in matched)
            broad_candidates.append({
                "candidate_index": index,
                "matched_target_count": len(matched),
                "candidate_bbox_area": area(candidate),
                "median_matched_target_area": median(area(target) for target in matched),
                "bbox_to_median_target_area": round(area(candidate) / max(1, median(area(target) for target in matched)), 2),
                "source_tile_count": len(candidate.get("source_tiles", [])),
            })
    overlap_pairs = [
        [first + 1, second + 1]
        for first in range(len(candidates))
        for second in range(first + 1, len(candidates))
        if intersection(candidates[first], candidates[second]) / max(1, min(area(candidates[first]), area(candidates[second]))) >= 0.5
    ]
    return {
        "image": case["image"],
        "kind": case["kind"],
        "candidate_count": len(candidates),
        "candidate_overlapping_target_count": candidate_hits,
        "candidate_without_target_count": len(candidates) - candidate_hits,
        "target_box_count": len(targets),
        "target_box_hit_count": sum(value > 0 for value in case["best_target_ious"]),
        "target_box_iou_ge_0_1": sum(value >= 0.1 for value in case["best_target_ious"]),
        "median_target_area": median(area(target) for target in targets) if targets else None,
        "median_hit_target_area": median(hit_target_areas) if hit_target_areas else None,
        "candidate_pairs_with_half_smaller_box_overlap": overlap_pairs,
        "broad_candidates": sorted(broad_candidates, key=lambda row: row["bbox_to_median_target_area"], reverse=True),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    if report["completed"] != report["planned"]:
        raise ValueError("input report is incomplete")
    cases = [summarize_case(case) for case in report["cases"] if case["kind"] != "normal"]
    output = {
        "source": str(args.input),
        "split": report["protocol"]["split"],
        "cases": cases,
        "summary": {
            "fault_images": len(cases),
            "candidate_count": sum(case["candidate_count"] for case in cases),
            "candidate_without_target_count": sum(case["candidate_without_target_count"] for case in cases),
            "target_box_count": sum(case["target_box_count"] for case in cases),
            "target_box_hit_count": sum(case["target_box_hit_count"] for case in cases),
            "candidate_pairs_with_half_smaller_box_overlap": sum(len(case["candidate_pairs_with_half_smaller_box_overlap"]) for case in cases),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))
    for case in sorted(cases, key=lambda row: row["candidate_without_target_count"], reverse=True)[:5]:
        print(case["image"], "nonoverlap", case["candidate_without_target_count"], "targets", case["target_box_hit_count"], "/", case["target_box_count"])


if __name__ == "__main__":
    main()
