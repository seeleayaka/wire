"""Audit label-free candidate postprocessing using an existing localization report.

The report contains labels for evaluation, but policies use only candidate
coordinates and scores. This is exploratory selection on val01, not a new
holdout or an automatic fault classifier. Truncating a saved report does not
reproduce changing the candidate budget inside the detector: the detector
uses a different internal ranking from its final display order.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def iou(candidate: dict, target: list[int]) -> float:
    width = max(0, min(candidate["right"], target[2]) - max(candidate["left"], target[0]))
    height = max(0, min(candidate["bottom"], target[3]) - max(candidate["top"], target[1]))
    intersection = width * height
    if not intersection:
        return 0.0
    candidate_area = (candidate["right"] - candidate["left"]) * (candidate["bottom"] - candidate["top"])
    target_area = (target[2] - target[0]) * (target[3] - target[1])
    return intersection / (candidate_area + target_area - intersection)


def resize(candidate: dict, scale: float) -> dict:
    center_x = (candidate["left"] + candidate["right"]) / 2
    center_y = (candidate["top"] + candidate["bottom"]) / 2
    half_width = (candidate["right"] - candidate["left"]) * scale / 2
    half_height = (candidate["bottom"] - candidate["top"]) * scale / 2
    return {
        **candidate,
        "left": round(center_x - half_width),
        "top": round(center_y - half_height),
        "right": round(center_x + half_width),
        "bottom": round(center_y + half_height),
    }


def select(candidates: list[dict], *, ranking: str, limit: int, scale: float) -> list[dict]:
    if ranking == "original":
        ranked = candidates
    elif ranking == "score":
        ranked = sorted(candidates, key=lambda item: item["difference_score"], reverse=True)
    elif ranking == "small_area":
        ranked = sorted(candidates, key=lambda item: item["area"])
    elif ranking == "score_density":
        ranked = sorted(candidates, key=lambda item: item["difference_score"] / max(item["area"], 1), reverse=True)
    else:
        raise ValueError(ranking)
    return [resize(item, scale) for item in ranked[:limit]]


def summarize(cases: list[dict], *, ranking: str, limit: int, scale: float) -> dict:
    target_ious = []
    candidate_hits = 0
    candidate_total = 0
    image_hits = 0
    for case in cases:
        candidates = select(case["candidates"], ranking=ranking, limit=limit, scale=scale)
        targets = case["target_boxes_aligned_xyxy"]
        best_ious = [max((iou(candidate, target) for candidate in candidates), default=0.0) for target in targets]
        target_ious.extend(best_ious)
        image_hits += any(value > 0 for value in best_ious)
        candidate_total += len(candidates)
        candidate_hits += sum(any(iou(candidate, target) > 0 for target in targets) for candidate in candidates)
    return {
        "ranking": ranking,
        "limit": limit,
        "scale": scale,
        "fault_images_with_target_overlap": image_hits,
        "fault_images": len(cases),
        "target_boxes_with_overlap": sum(value > 0 for value in target_ious),
        "target_boxes": len(target_ious),
        "target_boxes_iou_ge_0_1": sum(value >= 0.1 for value in target_ious),
        "target_boxes_iou_ge_0_5": sum(value >= 0.5 for value in target_ious),
        "mean_best_target_iou": sum(target_ious) / len(target_ious),
        "candidates_overlapping_target": candidate_hits,
        "candidate_count": candidate_total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    if source.get("completed") != source.get("planned"):
        raise ValueError("localization report is incomplete")
    cases = [case for case in source["cases"] if case["kind"] != "normal"]
    if any(len(case["target_boxes_aligned_xyxy"]) != case["source_target_box_count"] for case in cases):
        raise ValueError("target alignment is incomplete")
    results = [
        summarize(cases, ranking=ranking, limit=limit, scale=scale)
        for ranking in ("original", "score", "small_area", "score_density")
        for limit in (3, 4, 5, 6)
        for scale in (1.0, 0.8, 0.6)
    ]
    output = {
        "source": str(args.input),
        "split": source["protocol"]["split"],
        "warning": "Exploratory postprocessing audit only. Saved-report truncation is not equivalent to changing the detector budget. Select only on validation, never on reviewed test01.",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for result in sorted(results, key=lambda row: (-row["target_boxes_with_overlap"], -row["mean_best_target_iou"], row["candidate_count"]))[:10]:
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
