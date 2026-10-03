"""Save the existing DINO review heat and candidates for one Mendeley image.

This is an isolated diagnostic cache for local-refinement experiments. It does
not alter the GUI or any production candidate logic.
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

from evaluate_mendeley_balanced import FULL_REVIEW_ROI, read_image  # noqa: E402
from assembly_auto_review_dino_v2 import dino_fused_regions  # noqa: E402
import assembly_auto_review_robust_v3 as perspective  # noqa: E402
import tiled_dino_review  # noqa: E402


def decode_jet(heat: np.ndarray) -> np.ndarray:
    """Invert the exact uint8 OpenCV JET mapping used by dino_fused_regions."""
    palette = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(256, 1), cv2.COLORMAP_JET).reshape(256, 3)
    keys = palette[:, 0].astype(np.uint32) | (palette[:, 1].astype(np.uint32) << 8) | (palette[:, 2].astype(np.uint32) << 16)
    if len(np.unique(keys)) != 256:
        raise ValueError("JET palette is not invertible on this OpenCV version")
    lookup = np.zeros(1 << 24, dtype=np.uint8)
    lookup[keys] = np.arange(256, dtype=np.uint8)
    encoded = heat[:, :, 0].astype(np.uint32) | (heat[:, :, 1].astype(np.uint32) << 8) | (heat[:, :, 2].astype(np.uint32) << 16)
    return lookup[encoded]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--candidate-budget", type=int, default=6)
    args = parser.parse_args()
    if (args.image is None) == (args.image_dir is None):
        parser.error("provide exactly one of --image or --image-dir")
    tiled_dino_review.LARGE_ROI_MAX_CANDIDATES = args.candidate_budget
    reference = read_image(args.reference)
    images = [args.image] if args.image is not None else sorted(path for path in args.image_dir.glob("*.JPG") if not path.stem.startswith("normal_"))
    for index, image_path in enumerate(images, 1):
        output_prefix = args.output_prefix if args.image is not None else args.output_prefix / image_path.stem
        inspection = read_image(image_path)
        aligned, alignment = perspective.automatic_homography(reference, inspection)
        if aligned is None:
            print(json.dumps({"image": image_path.name, "alignment_failed": True, "alignment": alignment}), flush=True)
            continue
        _overlay, heat, candidates = dino_fused_regions(reference, aligned, FULL_REVIEW_ROI)
        score = decode_jet(heat)
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_prefix.with_suffix(".npz"), score=score)
        output_prefix.with_suffix(".json").write_text(json.dumps({
            "image": str(image_path),
            "reference": str(args.reference),
            "candidate_budget": args.candidate_budget,
            "image_shape": list(score.shape),
            "alignment": alignment,
            "candidates": candidates,
        }, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        if args.image is not None:
            ok, encoded = cv2.imencode(".jpg", heat)
            if ok:
                encoded.tofile(str(output_prefix.with_suffix(".jpg")))
        print(json.dumps({"progress": f"{index}/{len(images)}", "image": image_path.name, "candidates": len(candidates), "score_max": int(score.max()), "score_nonzero": int(np.count_nonzero(score))}), flush=True)


if __name__ == "__main__":
    main()
