"""Replay the generated cable-cabinet challenge set through the real DINO path.

The expected boxes below are evaluation annotations only.  They measure recall
and off-target candidate noise; they are never imported by the review pipeline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Preserve the Windows Torch-before-Qt import order used by the product entry.
from assembly_auto_review_dino_v2 import dino_fused_regions
import assembly_auto_review_robust as robust
import assembly_auto_review_robust_v3 as perspective


DEFAULT_FOLDER = Path(r"C:\Users\HUAWEI\Desktop\案例\线材柜子\1")
FULL_CABINET_ROI = [[12 / 1080, 30 / 712, 1029 / 1080, 695 / 712]]
ANNOTATIONS = {
    "wrong6.png": ("right_upper_unseated_wire", [940, 195, 1029, 275]),
    "wrong7.png": ("lower_CP1_CP5_missing_wire", [500, 500, 575, 610]),
    "wrong8.png": ("lower_right_terminal_miswire", [835, 500, 900, 610]),
    "wrong9.png": ("upper_CP10_CP13_double_and_missing", [435, 55, 475, 160]),
    "wrong10.png": ("right_upper_dangling_blue_tip", [820, 235, 920, 385]),
    "wrong11.png": ("lower_CP4_CP5_missing_wire", [480, 385, 555, 530]),
    "wrong12.png": ("lower_right_terminal_subtle_miswire", [770, 430, 860, 590]),
    "wrong13.png": ("upper_CP10_CP13_double_and_missing", [520, 145, 565, 225]),
}


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    extension = path.suffix or ".jpg"
    ok, encoded = cv2.imencode(extension, image)
    if not ok:
        raise ValueError(f"cannot encode {path}")
    encoded.tofile(str(path))


def overlaps(candidate: dict[str, Any], target: list[int]) -> bool:
    left = max(int(candidate["left"]), target[0])
    top = max(int(candidate["top"]), target[1])
    right = min(int(candidate["right"]), target[2])
    bottom = min(int(candidate["bottom"]), target[3])
    return right > left and bottom > top


def evaluate_case(reference: np.ndarray, inspection_path: Path, output: Path) -> dict[str, Any]:
    defect, target = ANNOTATIONS[inspection_path.name]
    inspection = read_image(inspection_path)
    aligned, alignment = perspective.automatic_homography(reference, inspection)
    result: dict[str, Any] = {
        "id": inspection_path.name,
        "expected_defect": defect,
        "expected_region": target,
        "alignment": alignment,
        "candidates": [],
    }
    if aligned is None:
        result.update(target_detected=False, off_target_candidate_count=0, alignment_failed=True)
        return result

    overlay, _heat, candidates = dino_fused_regions(reference, aligned, FULL_CABINET_ROI)
    output_candidates = []
    for candidate in candidates:
        item = {
            key: candidate[key]
            for key in (
                "left", "top", "right", "bottom", "area", "difference_score",
                "source_tiles", "evidence_summary", "evidence_scores",
            )
            if key in candidate
        }
        item["hits_expected_region"] = overlaps(candidate, target)
        output_candidates.append(item)
    result["candidates"] = output_candidates
    result["target_detected"] = any(item["hits_expected_region"] for item in output_candidates)
    result["off_target_candidate_count"] = sum(not item["hits_expected_region"] for item in output_candidates)
    result["local_alignment"] = robust.LAST_DIAGNOSTICS
    write_image(output / f"{inspection_path.stem}_aligned.jpg", aligned)
    write_image(output / f"{inspection_path.stem}_anomaly_boxes.jpg", overlay)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=DEFAULT_FOLDER)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    reference = read_image(args.folder / "right.png")
    cases = [evaluate_case(reference, args.folder / name, args.output) for name in ANNOTATIONS]
    report = {
        "mode": "real_dino_pipeline_full_cabinet_roi",
        "total": len(cases),
        "aligned": sum(not case.get("alignment_failed", False) for case in cases),
        "target_detected": sum(bool(case["target_detected"]) for case in cases),
        "candidate_count": sum(len(case["candidates"]) for case in cases),
        "off_target_candidate_count": sum(case["off_target_candidate_count"] for case in cases),
        "cases": cases,
    }
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("total", "aligned", "target_detected", "candidate_count", "off_target_candidate_count")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
