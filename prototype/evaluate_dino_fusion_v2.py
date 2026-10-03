"""Headless A/B runner for DINO fusion on one aligned review ROI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import assembly_auto_review_robust_v3 as perspective
from dino_feature_diff_v2 import fused_components


def read(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read {path}")
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("inspection", type=Path)
    parser.add_argument("--roi", nargs=4, type=int, required=True, metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference, inspection = read(args.reference), read(args.inspection)
    aligned, alignment = perspective.automatic_homography(reference, inspection)
    if aligned is None:
        raise SystemExit(json.dumps({"alignment": alignment}, ensure_ascii=False, indent=2))
    left, top, right, bottom = args.roi
    score, metadata, candidates = fused_components(reference[top:bottom, left:right], aligned[top:bottom, left:right])
    args.output.mkdir(parents=True, exist_ok=True)
    heat = cv2.applyColorMap(np.uint8(np.clip(score, 0, 255)), cv2.COLORMAP_JET)
    cv2.imencode(".jpg", heat)[1].tofile(str(args.output / "dino_fused_heatmap.jpg"))
    report = {"alignment": alignment, "roi": args.roi, "dino": metadata, "candidate_regions": candidates}
    (args.output / "dino_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
