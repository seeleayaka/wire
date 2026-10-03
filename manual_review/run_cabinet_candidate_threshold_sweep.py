"""Run the cabinet DINO mainline and compare read-only candidate evidence gates.

The production candidate extractor and GUI recipe are not changed.  Every base
``possible_difference_manual_review`` candidate is saved, then several numeric
post-filters are rendered and reported.  A rejected candidate is diagnostic
data, never silently deleted or treated as a known non-fault.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "prototype"))

# Import DINO/Torch before the later review modules reach a PyQt import path.
from assembly_auto_review_dino_v2 import dino_fused_regions  # noqa: E402
import assembly_auto_review_robust as robust  # noqa: E402
import assembly_auto_review_robust_v3 as perspective  # noqa: E402


FILTERS: tuple[dict[str, Any], ...] = (
    {"id": "base", "description": "No post-filter; saved mainline candidates."},
    {
        "id": "loose_dino014_cross010",
        "description": "DINO mean >= 0.14 and DINO/traditional cross evidence >= 0.10.",
        "dino_mean_min": 0.14,
        "cross_ratio_min": 0.10,
    },
    {
        "id": "balanced_dino016_cross016",
        "description": "Post-hoc set-2 candidate pre-gate: DINO mean >= 0.16 and cross evidence >= 0.16.",
        "dino_mean_min": 0.16,
        "cross_ratio_min": 0.16,
    },
    {
        "id": "medium_dino020_cross020",
        "description": "DINO mean >= 0.20 and cross evidence >= 0.20.",
        "dino_mean_min": 0.20,
        "cross_ratio_min": 0.20,
    },
    {
        "id": "strict_dino024_cross024",
        "description": "DINO mean >= 0.24 and cross evidence >= 0.24.",
        "dino_mean_min": 0.24,
        "cross_ratio_min": 0.24,
    },
    {
        "id": "score173_only",
        "description": "Deliberately score-only comparison: difference score >= 173.",
        "difference_score_min": 173.0,
    },
)


def parse_roi(value: str) -> list[float]:
    values = [float(item) for item in value.split(",")]
    if len(values) != 4 or not (0 <= values[0] < values[2] <= 1 and 0 <= values[1] < values[3] <= 1):
        raise argparse.ArgumentTypeError("ROI must be left,top,right,bottom in [0,1]")
    return values


def natural_key(path: Path) -> list[object]:
    return [int(item) if item.isdigit() else item.casefold() for item in re.split(r"(\d+)", path.name)]


def read_image(path: Path) -> np.ndarray:
    image = robust.auto.base.read_image(path)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def compact(candidate: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "check_region", "left", "top", "right", "bottom", "area", "difference_score", "confidence",
        "comparison_mode", "source_tiles", "evidence_summary", "evidence_scores",
    )
    return {key: candidate.get(key) for key in keys}


def evidence(candidate: dict[str, Any]) -> dict[str, float]:
    values = candidate.get("evidence_scores") or {}
    return {
        "difference_score": float(candidate.get("difference_score", 0.0) or 0.0),
        "dino_mean_max": float(values.get("dino_mean_max", 0.0) or 0.0),
        "cross_evidence_pixel_ratio_max": float(values.get("cross_evidence_pixel_ratio_max", 0.0) or 0.0),
    }


def matches(candidate: dict[str, Any], rule: dict[str, Any]) -> bool:
    values = evidence(candidate)
    return (
        values["difference_score"] >= float(rule.get("difference_score_min", float("-inf")))
        and values["dino_mean_max"] >= float(rule.get("dino_mean_min", float("-inf")))
        and values["cross_evidence_pixel_ratio_max"] >= float(rule.get("cross_ratio_min", float("-inf")))
    )


def render_filtered_overlay(aligned: np.ndarray, candidates: list[dict[str, Any]], kept: set[int], title: str) -> np.ndarray:
    overlay = aligned.copy()
    for index, candidate in enumerate(candidates, 1):
        if index not in kept:
            continue
        left, top, right, bottom = (int(candidate[key]) for key in ("left", "top", "right", "bottom"))
        values = evidence(candidate)
        cv2.rectangle(overlay, (left, top), (right, bottom), (0, 180, 255), 4)
        label = f"C{index} d={values['dino_mean_max']:.2f} x={values['cross_evidence_pixel_ratio_max']:.2f}"
        cv2.putText(overlay, label, (left, max(24, top - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 255), 2)
    cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 29), (32, 32, 32), thickness=-1)
    cv2.putText(overlay, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1)
    return overlay


def write_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Cannot write image: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--roi", type=parse_roi, default=[0.01, 0.02, 0.99, 0.98])
    parser.add_argument("--start-index", type=int, default=0, help="Zero-based index into the natural-sorted inspection images.")
    parser.add_argument("--max-images", type=int, help="Process at most this many images for resumable batches.")
    args = parser.parse_args()

    if not args.reference.is_file() or not args.directory.is_dir():
        raise FileNotFoundError("Reference or inspection directory does not exist")
    images = [
        path for path in args.directory.glob("*.png")
        if path.resolve() != args.reference.resolve()
    ]
    images.sort(key=natural_key)
    selected = images[args.start_index:]
    if args.max_images is not None:
        selected = selected[:args.max_images]
    if not selected:
        raise ValueError("No inspection images selected")

    args.output.mkdir(parents=True, exist_ok=True)
    reference = read_image(args.reference)
    cases: list[dict[str, Any]] = []
    for source_index, inspection_path in enumerate(selected, args.start_index + 1):
        inspection = read_image(inspection_path)
        aligned, alignment = perspective.automatic_homography(reference, inspection)
        case: dict[str, Any] = {
            "index": source_index,
            "image": str(inspection_path),
            "candidate_contract": "possible_difference_manual_review",
            "alignment": alignment,
            "base_candidates": [],
            "filters": {},
        }
        quality = alignment.get("alignment_quality", {})
        case["alignment_reliable"] = bool(quality.get("reliable", False))
        case["alignment_reason"] = quality.get("reason", alignment.get("reason"))
        if aligned is not None:
            _base_overlay, heat, candidates = dino_fused_regions(reference, aligned, [args.roi])
            case["base_candidates"] = [compact(candidate) for candidate in candidates]
            write_image(args.output / f"case_{source_index:02d}_heat.jpg", heat)
            for rule in FILTERS:
                kept = {index for index, candidate in enumerate(candidates, 1) if matches(candidate, rule)}
                suppressed = [index for index in range(1, len(candidates) + 1) if index not in kept]
                case["filters"][rule["id"]] = {
                    "rule": {key: value for key, value in rule.items() if key != "id"},
                    "kept_candidate_indices": sorted(kept),
                    "suppressed_candidate_indices": suppressed,
                    "kept_count": len(kept),
                    "suppressed_count": len(suppressed),
                }
                overlay = render_filtered_overlay(
                    aligned,
                    candidates,
                    kept,
                    f"case {source_index:02d} | {rule['id']} | kept={len(kept)}/{len(candidates)}",
                )
                write_image(args.output / f"case_{source_index:02d}_{rule['id']}_overlay.jpg", overlay)
        cases.append(case)
        print(
            json.dumps(
                {
                    "index": source_index,
                    "image": inspection_path.name,
                    "aligned": case["alignment_reliable"],
                    "base": len(case["base_candidates"]),
                    "balanced": case["filters"].get("balanced_dino016_cross016", {}).get("kept_count", 0),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    totals = {
        rule["id"]: sum(case["filters"].get(rule["id"], {}).get("kept_count", 0) for case in cases)
        for rule in FILTERS
    }
    report = {
        "kind": "cabinet_mainline_candidate_threshold_sweep",
        "boundary": "Diagnostic post-filter experiment only. Suppressed candidates remain in this report; this does not establish fault confidence, field accuracy, or electrical truth.",
        "reference": str(args.reference),
        "check_rois": [args.roi],
        "source_image_count": len(images),
        "selected_source_indices": [case["index"] for case in cases],
        "filters": FILTERS,
        "totals": totals,
        "cases": cases,
    }
    start = args.start_index + 1
    end = args.start_index + len(cases)
    report_path = args.output / f"report_part_{start:02d}_{end:02d}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "report": str(report_path), "totals": totals}, ensure_ascii=False))


if __name__ == "__main__":
    main()
