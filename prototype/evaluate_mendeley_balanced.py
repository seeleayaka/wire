"""Evaluate the real DINO review pipeline on a balanced Mendeley wire-fault sample.

The model never reads YOLO labels.  Labels are converted to pixel boxes only
after detection to report fault-image and labelled-fragment recall.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from assembly_auto_review_dino_v2 import dino_fused_regions
import assembly_auto_review_robust as robust
import assembly_auto_review_robust_v3 as perspective


DEFAULT_DATASET = Path(
    r"E:\PythonProject10\data\external_datasets\mendeley_electrical_wiring_faults"
    r"\Predictive Maintenance for Electrical Wiring Faults"
)
FULL_REVIEW_ROI = [[0.03, 0.04, 0.97, 0.96]]
FAULT_KINDS = ("damaged", "disconnected", "misrouted")


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read image: {path}")
    return image


def load_target_boxes(label_path: Path, width: int, height: int) -> list[list[int]]:
    boxes: list[list[int]] = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        _class_id, center_x, center_y, box_width, box_height = map(float, line.split())
        left = int(round((center_x - box_width / 2) * width))
        top = int(round((center_y - box_height / 2) * height))
        right = int(round((center_x + box_width / 2) * width))
        bottom = int(round((center_y + box_height / 2) * height))
        boxes.append([left, top, right, bottom])
    return boxes


def overlaps(candidate: dict[str, Any], target: list[int]) -> bool:
    left = max(int(candidate["left"]), target[0])
    top = max(int(candidate["top"]), target[1])
    right = min(int(candidate["right"]), target[2])
    bottom = min(int(candidate["bottom"]), target[3])
    return right > left and bottom > top


def choose_cases(image_folder: Path, reference_name: str, per_kind: int) -> list[Path]:
    images = sorted(image_folder.glob("*.JPG"))
    kind_of = lambda path: path.stem.split("_", 1)[0]
    selected = [path for path in images if kind_of(path) == "normal" and path.name != reference_name][:per_kind]
    for kind in FAULT_KINDS:
        selected.extend(path for path in images if kind_of(path) == kind)
        selected = selected[: per_kind + sum(per_kind for _ in FAULT_KINDS[: FAULT_KINDS.index(kind) + 1])]
    expected = per_kind * (1 + len(FAULT_KINDS))
    if len(selected) != expected:
        raise ValueError(f"expected {expected} cases in {image_folder}, got {len(selected)}")
    return selected


def evaluate_case(reference: np.ndarray, image_path: Path, label_folder: Path) -> dict[str, Any]:
    inspection = read_image(image_path)
    target_boxes = load_target_boxes(label_folder / f"{image_path.stem}.txt", inspection.shape[1], inspection.shape[0])
    aligned, alignment = perspective.automatic_homography(reference, inspection)
    result: dict[str, Any] = {
        "image": image_path.name,
        "kind": image_path.stem.split("_", 1)[0],
        "target_box_count": len(target_boxes),
        "alignment": alignment,
        "candidates": [],
    }
    if aligned is None:
        result.update(alignment_failed=True, target_box_hits=0, fault_image_hit=False)
        return result

    _overlay, _heat, candidates = dino_fused_regions(reference, aligned, FULL_REVIEW_ROI)
    result["candidates"] = [
        {
            key: candidate[key]
            for key in ("left", "top", "right", "bottom", "area", "difference_score", "source_tiles")
            if key in candidate
        }
        for candidate in candidates
    ]
    result["alignment_failed"] = False
    result["target_box_hits"] = sum(any(overlaps(candidate, target) for candidate in candidates) for target in target_boxes)
    result["fault_image_hit"] = bool(target_boxes) and result["target_box_hits"] > 0
    result["local_alignment"] = robust.LAST_DIAGNOSTICS
    return result


def build_summary(cases: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        groups[case["kind"]].append(case)
    summary: dict[str, dict[str, int]] = {}
    for kind, group in sorted(groups.items()):
        faults = [case for case in group if case["target_box_count"]]
        summary[kind] = {
            "images": len(group),
            "alignment_failures": sum(case["alignment_failed"] for case in group),
            "images_with_candidates": sum(bool(case["candidates"]) for case in group),
            "candidate_count": sum(len(case["candidates"]) for case in group),
            "target_boxes": sum(case["target_box_count"] for case in group),
            "target_box_hits": sum(case["target_box_hits"] for case in group),
            "fault_images": len(faults),
            "fault_images_hit": sum(case["fault_image_hit"] for case in faults),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-kind", type=int, default=5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    all_cases: list[dict[str, Any]] = []
    reference_by_split = {"test01": "normal_102.JPG", "train01": "normal_073.JPG"}
    planned: list[tuple[str, Path, np.ndarray, Path]] = []
    for split, reference_name in reference_by_split.items():
        image_folder = args.dataset / "images" / split
        label_folder = args.dataset / "labels" / split
        planned.extend((split, image_path, read_image(image_folder / reference_name), label_folder) for image_path in choose_cases(image_folder, reference_name, args.per_kind))

    for index, (split, image_path, reference, label_folder) in enumerate(planned, start=1):
        case = evaluate_case(reference, image_path, label_folder)
        case["split"] = split
        all_cases.append(case)
        progress = {"completed": len(all_cases), "total": len(planned), "cases": all_cases}
        (args.output / "progress.json").write_text(json.dumps(progress, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(
            f"[{index}/{len(planned)}] {split} {image_path.name} "
            f"candidates={len(case['candidates'])} hits={case['target_box_hits']}/{case['target_box_count']}",
            flush=True,
        )

    report = {
        "mode": "real_dino_pipeline_full_review_roi",
        "dataset": str(args.dataset),
        "protocol": {
            "splits": list(reference_by_split),
            "per_split": f"{args.per_kind} normal plus {args.per_kind} each damaged/disconnected/misrouted",
            "rois": FULL_REVIEW_ROI,
            "labels_used_only_for_evaluation": True,
        },
        "summary": {split: build_summary([case for case in all_cases if case["split"] == split]) for split in reference_by_split},
        "cases": all_cases,
    }
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
