"""Run the existing DINO/traditional candidate generator on one accepted alignment.

This is an independent diagnostic helper.  It does not modify mainline
alignment, candidate generation, model weights, or annotations.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "prototype"))
import assembly_auto_review_dino as dino  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--aligned-inspection", type=Path, required=True)
    parser.add_argument("--image-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return image


def main() -> int:
    args = parse_args()
    reference = read_image(args.reference)
    aligned = read_image(args.aligned_inspection)
    if reference.shape != aligned.shape:
        raise RuntimeError("Reference and accepted aligned inspection must share shape")
    _, _, candidates = dino.dino_fused_regions(reference, aligned, [[0.01, 0.02, 0.99, 0.98]])
    result = {
        "purpose": "Existing DINO/traditional visible-change candidates on a mainline-accepted aligned image.",
        "image": args.image_name,
        "reference": str(args.reference.resolve()),
        "aligned_inspection": str(args.aligned_inspection.resolve()),
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"candidate_count": len(candidates)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
