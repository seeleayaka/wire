"""Independent fixed-small-ROI probe for the two missed material-2 wire groups.

The ROI is a cabinet-coordinate preset, shared by both images.  It is not
drawn from either manual anomaly box and is not added to the mainline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cabinet_candidate_filter_sweep import (
    dino, perspective, read_image, replay_global_h, map_canvas_box_to_reference,
    rect_overlap, keep_candidate, overlay, write_json,
)

ROOT = Path(r"C:\Users\HUAWEI\Desktop\案例\线材柜子\2")
STANDARD = Path(__file__).with_name("cabinet_material2_qualitative_manual_standard_v1.json")
UPPER_MID_ROIS = {
    "wide": [0.45, 0.12, 0.77, 0.58],
    "medium": [0.50, 0.15, 0.75, 0.58],
    "tight": [0.55, 0.18, 0.73, 0.52],
}
TARGETS = {
    "ChatGPT Image 2026年8月25日 16_23_59 (1).png": {"Q01"},
    "ChatGPT Image 2026年8月25日 16_24_00 (2).png": {"Q01"},
}


def coverage(group: dict, candidates: list[dict]) -> dict:
    overlaps = [rect_overlap(group["mapped_reference_bbox"], candidate) for candidate in candidates]
    return {"id": group["id"], "proxy_coverage": any(bool(item["proxy_hit"]) for item in overlaps), "candidate_overlaps": overlaps}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("material2_upper_mid_roi_probe_20260827"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    standard = json.loads(STANDARD.read_text(encoding="utf-8"))
    reference = read_image(ROOT / "2.png")
    canvas_size = tuple(standard["coordinate_space"]["canvas_size"])
    result_cases = []
    for case in standard["images"]:
        if case["image"] not in TARGETS:
            continue
        inspection = read_image(ROOT / case["image"])
        aligned, alignment = perspective.automatic_homography(reference, inspection)
        if aligned is None:
            raise RuntimeError(f"Alignment rejected: {case['image']}")
        h = replay_global_h(reference, inspection)
        groups = []
        for group in case["groups"]:
            if group["id"] not in TARGETS[case["image"]]:
                continue
            groups.append({
                "id": group["id"], "manual_canvas_bbox": group["bbox"], "description": group["description"],
                "mapped_reference_bbox": map_canvas_box_to_reference(group["bbox"], (inspection.shape[1], inspection.shape[0]), h, canvas_size=canvas_size, destination_size=(reference.shape[1], reference.shape[0])),
            })
        variants = []
        for name, roi in UPPER_MID_ROIS.items():
            _, _, candidates = dino.dino_fused_regions(reference, aligned, [roi])
            guarded = [candidate for candidate in candidates if keep_candidate(candidate, 0.20, 0.20)]
            record = {
                "roi_name": name, "fixed_normalized_roi": roi, "candidates": candidates, "guarded_candidates": guarded,
                "coverage": [coverage(group, candidates) for group in groups],
                "guarded_coverage": [coverage(group, guarded) for group in groups],
            }
            variants.append(record)
            stem = Path(case["image"]).stem
            overlay(args.output / f"{stem}_{name}_baseline.png", aligned, groups, candidates, f"fixed {name} ROI; green=manual, magenta=candidate")
            overlay(args.output / f"{stem}_{name}_guarded.png", aligned, groups, guarded, f"fixed {name} ROI guarded; green=manual, magenta=candidate")
        report = {"image": case["image"], "alignment": alignment, "groups": groups, "variants": variants}
        result_cases.append(report)
        write_json(args.output / f"{Path(case['image']).stem}_report.json", report)
    result = {
        "purpose": "Independent fixed-zone scale probe. All three ROIs are shared cabinet-coordinate presets, not per-image manual boxes; no mainline source changes.",
        "targets": "Only the two previously missed upper-middle qualitative groups.",
        "cases": result_cases,
    }
    write_json(args.output / "summary.json", result)
    for case in result_cases:
        for variant in case["variants"]:
            hit = sum(item["proxy_coverage"] for item in variant["coverage"])
            guarded_hit = sum(item["proxy_coverage"] for item in variant["guarded_coverage"])
            print(case["image"], variant["roi_name"], f"baseline={hit}/{len(case['groups'])} candidates={len(variant['candidates'])}", f"guarded={guarded_hit}/{len(case['groups'])} candidates={len(variant['guarded_candidates'])}")


if __name__ == "__main__":
    main()
