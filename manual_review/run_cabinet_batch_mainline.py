"""Run the real cabinet DINO-review mainline over a chosen reference/image set.

This keeps batch experiments separate from the cabinet GUI recipe.  It does not
train a model or infer electrical faults; each output is a
``possible_difference_manual_review`` candidate set.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "prototype"))

# Import this wrapper first: it loads torch before the later PyQt import path.
from assembly_auto_review_dino_v2 import dino_fused_regions  # noqa: E402
import assembly_auto_review_robust as robust  # noqa: E402
import assembly_auto_review_robust_v3 as perspective  # noqa: E402


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def parse_roi(value: str) -> list[float]:
    values = [float(item) for item in value.split(",")]
    if len(values) != 4 or not (0 <= values[0] < values[2] <= 1 and 0 <= values[1] < values[3] <= 1):
        raise argparse.ArgumentTypeError("ROI must be left,top,right,bottom in [0,1]")
    return values


def compact(candidate: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "check_region", "left", "top", "right", "bottom", "area", "difference_score", "confidence",
        "comparison_mode", "source_tiles", "evidence_summary", "evidence_scores",
    )
    return {key: candidate.get(key) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--roi", type=parse_roi, default=[0.01, 0.02, 0.99, 0.98])
    parser.add_argument("--start-index", type=int, default=0, help="Zero-based offset into sorted inspection images.")
    parser.add_argument("--max-images", type=int, help="Process at most this many images; use for resumable batches.")
    args = parser.parse_args()

    if not args.reference.is_file():
        raise FileNotFoundError(args.reference)
    if not args.directory.is_dir():
        raise NotADirectoryError(args.directory)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    images = sorted(
        path for path in args.directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and path.resolve() != args.reference.resolve()
    )
    if not images:
        raise ValueError("No inspection images after excluding the reference")
    if args.start_index < 0 or args.start_index >= len(images):
        raise ValueError(f"--start-index must be in [0, {len(images) - 1}]")
    selected_images = images[args.start_index:]
    if args.max_images is not None:
        if args.max_images < 1:
            raise ValueError("--max-images must be positive")
        selected_images = selected_images[:args.max_images]

    args.output.mkdir(parents=True)
    reference = read_image(args.reference)
    cases: list[dict[str, Any]] = []
    for index, inspection_path in enumerate(selected_images, args.start_index + 1):
        inspection = read_image(inspection_path)
        aligned, alignment = perspective.automatic_homography(reference, inspection)
        case: dict[str, Any] = {
            "index": index,
            "image": str(inspection_path),
            "candidate_contract": "possible_difference_manual_review",
            "alignment": alignment,
            "candidates": [],
        }
        if aligned is not None:
            overlay, heat, candidates = dino_fused_regions(reference, aligned, [args.roi])
            case["candidates"] = [compact(candidate) for candidate in candidates]
            case["local_alignment"] = robust.LAST_DIAGNOSTICS
            if not cv2.imwrite(str(args.output / f"case_{index:02d}_candidate_overlay.jpg"), overlay):
                raise RuntimeError("Cannot write candidate overlay")
            if not cv2.imwrite(str(args.output / f"case_{index:02d}_heat.jpg"), heat):
                raise RuntimeError("Cannot write heat image")
        quality = alignment.get("alignment_quality", {})
        case["alignment_reliable"] = bool(quality.get("reliable", False))
        case["alignment_reason"] = quality.get("reason", alignment.get("reason"))
        case["candidate_count"] = len(case["candidates"])
        cases.append(case)
        print(json.dumps({"index": index, "image": inspection_path.name, "aligned": case["alignment_reliable"], "candidate_count": case["candidate_count"]}, ensure_ascii=False), flush=True)

    report = {
        "kind": "cabinet_real_mainline_batch",
        "reference": str(args.reference),
        "reference_size": [int(reference.shape[1]), int(reference.shape[0])],
        "check_rois": [args.roi],
        "source_image_count": len(images),
        "selected_source_indices": [item["index"] for item in cases],
        "pipeline": "automatic_homography -> full_roi -> DINO_traditional_fusion -> possible_difference_manual_review",
        "boundary": "Visual smoke cases only; not field accuracy or electrical-fault truth.",
        "cases": cases,
    }
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "total": len(cases), "aligned": sum(item["alignment_reliable"] for item in cases), "candidate_count": sum(item["candidate_count"] for item in cases)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
