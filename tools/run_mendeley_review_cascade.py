"""Run the frozen normal-bank gate and visual candidate review on one image.

This entry point is scoped to the fixed Mendeley chassis. A normal-like image
does not consume the expensive DINO localization pass; a suspicious image
keeps the existing DINO/traditional candidates for manual review.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "prototype"))

from dino_feature_diff import extract_features  # noqa: E402
from inspection_agent.normal_reference import descriptor_from_patch_features, nearest_normal_score  # noqa: E402
from assembly_auto_review_dino_v2 import dino_fused_regions  # noqa: E402
import assembly_auto_review_robust_v3 as perspective  # noqa: E402
from evaluate_mendeley_balanced import FULL_REVIEW_ROI  # noqa: E402
import tiled_dino_review  # noqa: E402


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read image: {path}")
    return image


def review_image(image_path: Path, model_path: Path, train_normal_dir: Path, *, reference_mode: str = "fixed", fixed_reference: str = "normal_073.JPG", candidate_budget: int = 6) -> dict:
    if reference_mode not in {"fixed", "nearest"}:
        raise ValueError("reference_mode must be fixed or nearest")
    if candidate_budget < 1:
        raise ValueError("candidate_budget must be positive")
    tiled_dino_review.LARGE_ROI_MAX_CANDIDATES = candidate_budget
    saved = np.load(model_path)
    if int(saved["schema_version"]) != 1:
        raise ValueError("normal-bank model schema_version must be 1")
    bank = saved["descriptors"].astype(np.float32)
    reference_names = [str(name) for name in saved["reference_names"]]
    threshold = float(saved["threshold"])
    k = int(saved["k"])
    image = read_image(image_path)
    features, _metadata = extract_features(image, cache_reference=False)
    descriptor = descriptor_from_patch_features(features, grid_size=2)
    score, neighbors = nearest_normal_score(descriptor, bank, k=k)
    nearest_name = reference_names[neighbors[0]["index"]]
    for neighbor in neighbors:
        neighbor["image"] = reference_names[neighbor.pop("index")]
    possible_fault = score >= threshold
    report = {
        "schema_version": 1,
        "image": str(image_path),
        "normal_bank_score": score,
        "threshold": threshold,
        "decision": "possible_fault_manual_review" if possible_fault else "normal_like_reference_bank",
        "automatic_fault_verdict": False,
        "nearest_normal_references": neighbors,
        "reference_mode": reference_mode,
        "candidate_budget": candidate_budget,
        "localization_status": "skipped_normal_like_reference_bank",
        "candidates": [],
        "evidence_boundary": "Mendeley chassis image-level triage and manual-review candidates only",
    }
    if not possible_fault:
        return report
    reference_name = nearest_name if reference_mode == "nearest" else fixed_reference
    reference = read_image(train_normal_dir / reference_name)
    aligned, alignment = perspective.automatic_homography(reference, image)
    report["reference"] = reference_name
    report["alignment"] = alignment
    if aligned is None:
        report["localization_status"] = "alignment_failed_manual_review"
        return report
    _overlay, _heat, candidates = dino_fused_regions(reference, aligned, FULL_REVIEW_ROI)
    report["candidates"] = [
        {key: value for key, value in candidate.items() if key in {"left", "top", "right", "bottom", "area", "difference_score", "source_tiles", "evidence_summary", "evidence_scores"}}
        for candidate in candidates
    ]
    report["localization_status"] = "possible_difference_manual_review"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-normal-dir", type=Path, required=True)
    parser.add_argument("--reference-mode", choices=["fixed", "nearest"], default="fixed")
    parser.add_argument("--fixed-reference", default="normal_073.JPG")
    parser.add_argument("--candidate-budget", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = review_image(args.image, args.model, args.train_normal_dir, reference_mode=args.reference_mode, fixed_reference=args.fixed_reference, candidate_budget=args.candidate_budget)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": report["decision"], "localization_status": report["localization_status"], "candidate_count": len(report["candidates"]), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
