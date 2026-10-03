"""Keep DINO change regions that contain aligned SAM3 wire-mask change evidence.

The output boxes are local *visible wire-related change regions*.  A box does
not claim a complete physical cable, a terminal identity, or a fault type.
SAM3 supplies wire-related support; the existing DINO/traditional candidate
supplies independent visible-change evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sam-report", type=Path, required=True)
    parser.add_argument("--sam-inspection-only-mask", type=Path, required=True)
    parser.add_argument("--sam-reference-only-mask", type=Path, required=True)
    parser.add_argument("--dino-report", type=Path, required=True)
    parser.add_argument("--aligned-image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-sam-support-pixels", type=int, default=120)
    return parser.parse_args()


def read_image(path: Path, flags: int) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, flags)
    if image is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise RuntimeError(f"Cannot encode image: {path}")
    encoded.tofile(str(path))


def components_inside(mask: np.ndarray, min_pixels: int = 30) -> int:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return sum(int(stats[label, cv2.CC_STAT_AREA]) >= min_pixels for label in range(1, count))


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sam_report = json.loads(args.sam_report.read_text(encoding="utf-8"))
    dino_report = json.loads(args.dino_report.read_text(encoding="utf-8"))
    aligned = read_image(args.aligned_image, cv2.IMREAD_COLOR)
    added = read_image(args.sam_inspection_only_mask, cv2.IMREAD_GRAYSCALE) > 0
    missing = read_image(args.sam_reference_only_mask, cv2.IMREAD_GRAYSCALE) > 0
    if aligned.shape[:2] != added.shape or added.shape != missing.shape:
        raise RuntimeError("Aligned image and both SAM evidence masks must share dimensions")

    height, width = added.shape
    kept: list[dict[str, Any]] = []
    drawing = aligned.astype(np.float32).copy()
    drawing[added] = drawing[added] * 0.55 + np.array([0, 80, 255]) * 0.45
    drawing[missing] = drawing[missing] * 0.55 + np.array([255, 160, 0]) * 0.45
    drawing = np.clip(drawing, 0, 255).astype(np.uint8)

    for source_index, candidate in enumerate(dino_report.get("candidates", []), start=1):
        left = int(np.clip(candidate["left"], 0, width))
        top = int(np.clip(candidate["top"], 0, height))
        right = int(np.clip(candidate["right"], 0, width))
        bottom = int(np.clip(candidate["bottom"], 0, height))
        if right <= left or bottom <= top:
            continue
        added_crop = added[top:bottom, left:right]
        missing_crop = missing[top:bottom, left:right]
        added_pixels = int(added_crop.sum())
        missing_pixels = int(missing_crop.sum())
        support_pixels = added_pixels + missing_pixels
        if support_pixels < args.min_sam_support_pixels:
            continue
        record = {
            "id": f"wire_change_{len(kept) + 1:02d}",
            "bbox_xyxy": [left, top, right, bottom],
            "source_dino_candidate_index": source_index,
            "sam_added_pixels": added_pixels,
            "sam_missing_pixels": missing_pixels,
            "sam_support_pixels": support_pixels,
            "sam_added_component_count": components_inside(added_crop),
            "sam_missing_component_count": components_inside(missing_crop),
            "dino_evidence_scores": candidate.get("evidence_scores", {}),
            "semantic": "visible_wire_related_change_region",
            "not_claimed": ["physical_cable_identity", "terminal_assignment", "fault_type"],
        }
        kept.append(record)
        cv2.rectangle(drawing, (left, top), (right, bottom), (0, 220, 0), 2)
        cv2.putText(drawing, record["id"], (left, max(18, top - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 220, 0), 2, cv2.LINE_AA)

    legend = "green: DINO change region with SAM wire support; red/+ added, blue/- missing"
    cv2.putText(drawing, legend, (15, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(drawing, legend, (15, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (20, 20, 20), 1, cv2.LINE_AA)
    write_image(args.output_dir / "visible_wire_change_fusion.jpg", drawing)
    result = {
        "purpose": "Independent SAM3 + existing DINO/traditional candidate fusion for visible wire-related change regions.",
        "inputs": {
            "sam_report": str(args.sam_report.resolve()),
            "dino_report": str(args.dino_report.resolve()),
            "aligned_image": str(args.aligned_image.resolve()),
        },
        "rules": {
            "requires_existing_dino_traditional_candidate": True,
            "min_sam_support_pixels": args.min_sam_support_pixels,
            "sam_masks_already_aligned": bool(sam_report.get("rules", {}).get("already_aligned")),
            "no_mask_joining": True,
        },
        "input_dino_candidate_count": len(dino_report.get("candidates", [])),
        "visible_wire_change_region_count": len(kept),
        "regions": kept,
    }
    (args.output_dir / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"visible_wire_change_region_count": len(kept)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
