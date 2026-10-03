"""Compare fixed versus nearest-train-normal localization on one dataset split.

Reference selection uses image descriptors only. Labels enter only after the
existing DINO/traditional candidate pipeline has finished inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "prototype"))

from evaluate_mendeley_balanced import evaluate_case, load_target_boxes, read_image  # noqa: E402
from tools.evaluate_mendeley_holdout_current import estimate_transform, transform_boxes  # noqa: E402
import tiled_dino_review  # noqa: E402


def intersection_over_union(candidate: dict, target: list[int]) -> float:
    left = max(candidate["left"], target[0])
    top = max(candidate["top"], target[1])
    right = min(candidate["right"], target[2])
    bottom = min(candidate["bottom"], target[3])
    intersect = max(0, right - left) * max(0, bottom - top)
    candidate_area = max(0, candidate["right"] - candidate["left"]) * max(0, candidate["bottom"] - candidate["top"])
    target_area = max(0, target[2] - target[0]) * max(0, target[3] - target[1])
    return intersect / (candidate_area + target_area - intersect) if intersect else 0.0


def summarize(records: list[dict]) -> dict:
    faults = [x for x in records if x["kind"] != "normal"]
    normals = [x for x in records if x["kind"] == "normal"]
    ious = [value for row in faults for value in row["best_target_ious"]]
    return {
        "images": len(records),
        "fault_images": len(faults),
        "normal_images": len(normals),
        "alignment_failures": sum(x["alignment_failed"] for x in records),
        "fault_images_with_candidate": sum(bool(x["candidates"]) for x in faults),
        "normal_images_with_candidate": sum(bool(x["candidates"]) for x in normals),
        "fault_images_with_target_overlap": sum(any(value > 0 for value in x["best_target_ious"]) for x in faults),
        "target_boxes": len(ious),
        "target_boxes_with_overlap": sum(value > 0 for value in ious),
        "target_boxes_iou_ge_0_1": sum(value >= 0.1 for value in ious),
        "target_boxes_iou_ge_0_5": sum(value >= 0.5 for value in ious),
        "mean_best_target_iou": sum(ious) / len(ious) if ious else None,
        "total_candidates": sum(len(x["candidates"]) for x in records),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--bank-report", type=Path, required=True)
    parser.add_argument("--split", choices=["val01", "test01"], default="val01")
    parser.add_argument("--reference-mode", choices=["fixed", "nearest"], required=True)
    parser.add_argument("--fixed-reference", default="normal_073.JPG")
    parser.add_argument("--fault-only", action="store_true", help="Evaluate localization on fault images only")
    parser.add_argument("--candidate-budget", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.candidate_budget < 1:
        raise ValueError("candidate-budget must be positive")
    tiled_dino_review.LARGE_ROI_MAX_CANDIDATES = args.candidate_budget
    bank_report = json.loads(args.bank_report.read_text(encoding="utf-8"))
    bank_records = {x["image"]: x for x in bank_report["validation" if args.split == "val01" else "test"]}
    image_dir = args.dataset / "images" / args.split
    train_dir = args.dataset / "images" / "train01"
    label_dir = args.dataset / "labels" / args.split
    planned = sorted(image_dir.glob("*.JPG"))
    if args.fault_only:
        planned = [path for path in planned if not path.stem.startswith("normal_")]
    records = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index, image_path in enumerate(planned, 1):
        if args.reference_mode == "nearest":
            reference_name = bank_records[image_path.name]["nearest_normal_references"][0]["image"]
        else:
            reference_name = args.fixed_reference
        reference_path = train_dir / reference_name
        reference = read_image(reference_path)
        inspection = read_image(image_path)
        case = evaluate_case(reference, image_path, label_dir)
        transform = estimate_transform(reference, inspection)
        raw_targets = load_target_boxes(label_dir / f"{image_path.stem}.txt", inspection.shape[1], inspection.shape[0])
        if transform is not None and not case["alignment_failed"]:
            targets = transform_boxes(raw_targets, transform)
            ious = [max((intersection_over_union(candidate, target) for candidate in case["candidates"]), default=0.0) for target in targets]
        else:
            targets = []
            ious = [0.0] * len(raw_targets)
        record = {
            "image": image_path.name,
            "kind": case["kind"],
            "reference": reference_name,
            "alignment_failed": case["alignment_failed"] or transform is None,
            "candidates": case["candidates"],
            "source_target_box_count": len(raw_targets),
            "target_boxes_aligned_xyxy": targets,
            "best_target_ious": ious,
        }
        records.append(record)
        report = {
            "protocol": {"split": args.split, "reference_mode": args.reference_mode, "reference_source": "train01 normals", "candidate_budget": args.candidate_budget, "labels_used_after_inference": True},
            "completed": len(records),
            "planned": len(planned),
            "summary": summarize(records),
            "cases": records,
        }
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[{index}/{len(planned)}] {image_path.name} ref={reference_name} candidates={len(case['candidates'])} mean_iou={(sum(ious)/len(ious) if ious else 0):.4f}", flush=True)
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
