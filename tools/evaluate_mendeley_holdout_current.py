"""Run the current DINO review pipeline on the complete Mendeley test split.

One normal test image is used as the fixed reference.  Every other test image
is evaluated without parameter tuning.  Source boxes are used only after
inference and are transformed into the aligned reference coordinates before
overlap scoring.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT = Path(r"E:\PythonProject10")
PROTOTYPE = PROJECT / "prototype"
sys.path.insert(0, str(PROTOTYPE))

from evaluate_mendeley_balanced import (  # noqa: E402
    FULL_REVIEW_ROI,
    build_summary,
    evaluate_case,
    load_target_boxes,
    overlaps,
    read_image,
)

DEFAULT_DATASET = (
    PROJECT
    / "data"
    / "external_datasets"
    / "mendeley_electrical_wiring_faults"
    / "Predictive Maintenance for Electrical Wiring Faults"
)


def estimate_transform(reference: np.ndarray, inspection: np.ndarray) -> np.ndarray | None:
    """Reproduce the production SIFT/MAGSAC transform for label coordinates."""
    sift = cv2.SIFT_create(nfeatures=9000, contrastThreshold=0.014, edgeThreshold=12)
    ref_keypoints, ref_descriptors = sift.detectAndCompute(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), None)
    test_keypoints, test_descriptors = sift.detectAndCompute(cv2.cvtColor(inspection, cv2.COLOR_BGR2GRAY), None)
    if ref_descriptors is None or test_descriptors is None:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(test_descriptors, ref_descriptors, k=2)
    good = [first for first, second in pairs if first.distance < 0.70 * second.distance]
    if len(good) < 60:
        return None
    source = np.float32([test_keypoints[match.queryIdx].pt for match in good]).reshape(-1, 1, 2)
    destination = np.float32([ref_keypoints[match.trainIdx].pt for match in good]).reshape(-1, 1, 2)
    method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
    transform, mask = cv2.findHomography(
        source,
        destination,
        method=method,
        ransacReprojThreshold=4.0,
        maxIters=10000,
        confidence=0.995,
    )
    return transform if transform is not None and mask is not None else None


def transform_boxes(boxes: list[list[int]], transform: np.ndarray) -> list[list[int]]:
    transformed: list[list[int]] = []
    for left, top, right, bottom in boxes:
        corners = np.float32([[[left, top], [right, top], [right, bottom], [left, bottom]]])
        warped = cv2.perspectiveTransform(corners, transform).reshape(-1, 2)
        transformed.append(
            [
                int(np.floor(warped[:, 0].min())),
                int(np.floor(warped[:, 1].min())),
                int(np.ceil(warped[:, 0].max())),
                int(np.ceil(warped[:, 1].max())),
            ]
        )
    return transformed


def save_overlay(
    output_path: Path,
    aligned: np.ndarray,
    candidates: list[dict[str, Any]],
    target_boxes: list[list[int]],
) -> None:
    canvas = aligned.copy()
    for number, candidate in enumerate(candidates, 1):
        bounds = tuple(int(candidate[key]) for key in ("left", "top", "right", "bottom"))
        cv2.rectangle(canvas, bounds[:2], bounds[2:], (0, 165, 255), 5)
        cv2.putText(canvas, f"C{number}", (bounds[0], max(35, bounds[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 3)
    for number, (left, top, right, bottom) in enumerate(target_boxes, 1):
        cv2.rectangle(canvas, (left, top), (right, bottom), (0, 220, 0), 3)
        cv2.putText(canvas, f"T{number}", (left, min(canvas.shape[0] - 8, bottom + 28)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 170, 0), 2)
    ok, encoded = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 91])
    if not ok:
        raise RuntimeError(f"cannot encode overlay: {output_path}")
    encoded.tofile(str(output_path))


def safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def overall_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    normals = [case for case in cases if case["kind"] == "normal"]
    faults = [case for case in cases if case["kind"] != "normal"]
    aligned_faults = [case for case in faults if not case["alignment_failed"]]
    normal_with_candidates = sum(bool(case["candidates"]) for case in normals if not case["alignment_failed"])
    fault_with_candidates = sum(bool(case["candidates"]) for case in aligned_faults)
    fault_hits = sum(bool(case["fault_image_hit"]) for case in aligned_faults)
    target_boxes = sum(case["target_box_count"] for case in aligned_faults)
    target_hits = sum(case["target_box_hits"] for case in aligned_faults)
    aligned_normals = sum(not case["alignment_failed"] for case in normals)
    return {
        "evaluated_images": len(cases),
        "normal_images": len(normals),
        "fault_images": len(faults),
        "alignment_failures": sum(case["alignment_failed"] for case in cases),
        "normal_images_with_candidates": normal_with_candidates,
        "normal_image_specificity": safe_ratio(aligned_normals - normal_with_candidates, aligned_normals),
        "fault_images_with_candidates": fault_with_candidates,
        "fault_image_candidate_sensitivity": safe_ratio(fault_with_candidates, len(aligned_faults)),
        "fault_images_with_localized_box_hit": fault_hits,
        "fault_image_localization_recall": safe_ratio(fault_hits, len(aligned_faults)),
        "source_target_boxes": target_boxes,
        "source_target_box_hits": target_hits,
        "source_target_box_recall": safe_ratio(target_hits, target_boxes),
        "candidate_count": sum(len(case["candidates"]) for case in cases),
        "mean_candidates_per_aligned_image": safe_ratio(
            sum(len(case["candidates"]) for case in cases),
            sum(not case["alignment_failed"] for case in cases),
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", default="normal_102.JPG")
    args = parser.parse_args()

    image_dir = args.dataset / "images" / "test01"
    label_dir = args.dataset / "labels" / "test01"
    output = args.output
    overlay_dir = output / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    reference_path = image_dir / args.reference
    reference = read_image(reference_path)
    planned = [path for path in sorted(image_dir.glob("*.JPG")) if path.name != args.reference]
    cases: list[dict[str, Any]] = []

    for index, image_path in enumerate(planned, 1):
        case = evaluate_case(reference, image_path, label_dir)
        case["split"] = "test01"
        inspection = read_image(image_path)
        transform = estimate_transform(reference, inspection)
        if not case["alignment_failed"] and transform is not None:
            raw_targets = load_target_boxes(label_dir / f"{image_path.stem}.txt", inspection.shape[1], inspection.shape[0])
            aligned_targets = transform_boxes(raw_targets, transform)
            case["target_boxes_aligned_xyxy"] = aligned_targets
            case["target_box_hits"] = sum(
                any(overlaps(candidate, target) for candidate in case["candidates"])
                for target in aligned_targets
            )
            case["fault_image_hit"] = bool(aligned_targets) and case["target_box_hits"] > 0
            aligned = cv2.warpPerspective(
                inspection,
                transform,
                (reference.shape[1], reference.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            save_overlay(overlay_dir / f"{image_path.stem}.jpg", aligned, case["candidates"], aligned_targets)
        else:
            case["target_boxes_aligned_xyxy"] = []
        cases.append(case)
        partial = {
            "schema_version": 1,
            "status": "running",
            "reference": str(reference_path),
            "completed": len(cases),
            "total": len(planned),
            "summary": overall_summary(cases),
            "cases": cases,
        }
        (output / "progress.json").write_text(json.dumps(partial, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(
            f"[{index}/{len(planned)}] {image_path.name} "
            f"candidates={len(case['candidates'])} hits={case['target_box_hits']}/{case['target_box_count']}",
            flush=True,
        )

    report = {
        "schema_version": 1,
        "status": "ok",
        "mode": "current_dino_pipeline_complete_mendeley_test01_holdout",
        "dataset": str(args.dataset),
        "protocol": {
            "split": "test01",
            "reference": args.reference,
            "evaluated": "all other test01 images",
            "review_roi": FULL_REVIEW_ROI,
            "labels_used_only_after_inference": True,
            "source_boxes_transformed_to_aligned_coordinates": True,
            "no_parameter_tuning": True,
        },
        "overall": overall_summary(cases),
        "by_kind": build_summary(cases),
        "cases": cases,
        "evidence_boundary": [
            "Source boxes localize documented component or fault regions; they are not per-cable masks.",
            "Candidate overlap is review evidence, not cable identity, endpoint attribution, electrical continuity, or topology accuracy.",
            "All images come from one fixed Dell chassis under controlled lighting; results are not field accuracy.",
        ],
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["overall"], ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
