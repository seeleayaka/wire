"""Evaluate heat-map connected components inside existing review candidates.

This is a validation-only experiment. It never uses labels to create or rank a
candidate; labels are read only for the final metrics. The original broad box
can optionally remain as a separate fallback review tier.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def box_area(box: dict | list) -> int:
    values = [box[key] for key in ("left", "top", "right", "bottom")] if isinstance(box, dict) else box
    return max(0, values[2] - values[0]) * max(0, values[3] - values[1])


def iou(candidate: dict, target: list[int]) -> float:
    width = max(0, min(candidate["right"], target[2]) - max(candidate["left"], target[0]))
    height = max(0, min(candidate["bottom"], target[3]) - max(candidate["top"], target[1]))
    intersection = width * height
    if not intersection:
        return 0.0
    return intersection / (box_area(candidate) + box_area(target) - intersection)


def refine_one(candidate: dict, score: np.ndarray, percentile: int, minimum: int, min_pixels: int, margin: int) -> dict:
    left, top, right, bottom = (int(candidate[key]) for key in ("left", "top", "right", "bottom"))
    crop = score[max(0, top):min(score.shape[0], bottom), max(0, left):min(score.shape[1], right)]
    if crop.size == 0:
        return dict(candidate)
    threshold = max(minimum, int(np.percentile(crop, percentile)))
    mask = (crop >= threshold).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    choices = []
    for index in range(1, count):
        x, y, width, height, pixels = (int(value) for value in stats[index])
        if pixels < min_pixels:
            continue
        mean_score = float(score[top + y:top + y + height, left + x:left + x + width][labels[y:y + height, x:x + width] == index].mean())
        choices.append((pixels * mean_score, x, y, width, height, pixels))
    if not choices:
        return dict(candidate)
    _rank, x, y, width, height, pixels = max(choices)
    return {
        **candidate,
        "left": max(0, left + x - margin),
        "top": max(0, top + y - margin),
        "right": min(score.shape[1], left + x + width + margin),
        "bottom": min(score.shape[0], top + y + height + margin),
        "heat_component_pixels": pixels,
        "heat_threshold": threshold,
    }


def summarize(cases: list[dict], candidate_key: str) -> dict:
    all_ious = []
    candidate_hits = 0
    candidate_total = 0
    image_hits = 0
    for case in cases:
        candidates = case[candidate_key]
        targets = case["targets"]
        ious = [max((iou(candidate, target) for candidate in candidates), default=0.0) for target in targets]
        all_ious.extend(ious)
        image_hits += any(value > 0 for value in ious)
        candidate_total += len(candidates)
        candidate_hits += sum(any(iou(candidate, target) > 0 for target in targets) for candidate in candidates)
    return {
        "fault_images_with_target_overlap": image_hits,
        "fault_images": len(cases),
        "target_boxes_with_overlap": sum(value > 0 for value in all_ious),
        "target_boxes": len(all_ious),
        "target_boxes_iou_ge_0_1": sum(value >= 0.1 for value in all_ious),
        "target_boxes_iou_ge_0_5": sum(value >= 0.5 for value in all_ious),
        "mean_best_target_iou": sum(all_ious) / len(all_ious),
        "candidate_overlapping_target_count": candidate_hits,
        "candidate_count": candidate_total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if report["protocol"]["split"] != "val01":
        raise ValueError("this experiment accepts val01 only")
    cases = []
    for source_case in report["cases"]:
        stem = Path(source_case["image"]).stem
        cache = json.loads((args.cache_dir / f"{stem}.json").read_text(encoding="utf-8"))
        with np.load(args.cache_dir / f"{stem}.npz") as data:
            score = data["score"]
        candidates = cache["candidates"]
        targets = source_case["target_boxes_aligned_xyxy"]
        cases.append({"image": source_case["image"], "candidates": candidates, "targets": targets, "score": score})
    baseline = summarize(cases, "candidates")
    expected = report["summary"]
    if (
        baseline["target_boxes_with_overlap"] != expected["target_boxes_with_overlap"]
        or baseline["candidate_count"] != expected["total_candidates"]
        or abs(baseline["mean_best_target_iou"] - expected["mean_best_target_iou"]) > 1e-6
    ):
        raise ValueError(f"cached inference differs from source report: {baseline} versus {expected}")
    variants = []
    for percentile in (65, 75, 85, 92, 97):
        for minimum in (120, 160, 200):
            for min_pixels in (80, 400):
                prepared = []
                changed = 0
                for case in cases:
                    refined = [refine_one(candidate, case["score"], percentile, minimum, min_pixels, margin=12) for candidate in case["candidates"]]
                    changed += sum(box_area(a) < box_area(b) for a, b in zip(refined, case["candidates"]))
                    prepared.append({"targets": case["targets"], "refined": refined, "with_fallback": case["candidates"] + refined})
                variants.append({
                    "percentile": percentile,
                    "minimum": minimum,
                    "min_pixels": min_pixels,
                    "changed_candidates": changed,
                    "refined_only": summarize(prepared, "refined"),
                    "with_fallback": summarize(prepared, "with_fallback"),
                })
    output = {"source_report": str(args.report), "cache_dir": str(args.cache_dir), "baseline": baseline, "variants": variants}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"baseline": baseline, "top_by_refined_iou": sorted(variants, key=lambda row: row["refined_only"]["mean_best_target_iou"], reverse=True)[:3]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
