"""Prepare a mainline-aligned cabinet image before SAM3 inference.

This helper invokes the existing SIFT/MAGSAC acceptance-gated alignment rather
than reimplementing it.  It writes a new aligned image and report only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Import Torch before the PyQt-owning review modules on this Windows setup.
import torch  # noqa: F401
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "prototype"))
import assembly_auto_review_robust_v3 as perspective  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--inspection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError(f"Cannot encode image: {path}")
    encoded.tofile(str(path))


def main() -> int:
    args = parse_args()
    if not args.reference.is_file() or not args.inspection.is_file():
        raise FileNotFoundError("Both --reference and --inspection must be image files")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference = read_image(args.reference)
    inspection = read_image(args.inspection)
    aligned, report = perspective.automatic_homography(reference, inspection)
    result = {
        "purpose": "Prepare an acceptance-gated global-homography image for SAM3 input.",
        "reference": str(args.reference.resolve()),
        "inspection": str(args.inspection.resolve()),
        "alignment": report,
        "aligned": aligned is not None,
        "aligned_size_wh": [int(reference.shape[1]), int(reference.shape[0])],
    }
    if aligned is not None:
        write_image(args.output_dir / "aligned_input.jpg", aligned)
        valid = perspective.auto.LAST_WARP_VALID_MASK
        if isinstance(valid, np.ndarray):
            write_image(args.output_dir / "valid_warp_mask.jpg", valid)
    (args.output_dir / "alignment_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"aligned": result["aligned"], "alignment": report}, ensure_ascii=False))
    return 0 if aligned is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
