"""Independent high-resolution bright thin-line residual probe.

It tests whether small loose white conductors can be surfaced without using
manual boxes as input.  It deliberately rejects broad/horizontal residuals;
it is not part of the mainline and produces review candidates only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch  # must precede PyQt-owning perspective import on this machine
import cv2
import numpy as np

ROOT = Path(r"C:\Users\HUAWEI\Desktop\案例\线材柜子\2")
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "prototype"))
from assembly_auto_review_robust_v3 import automatic_homography  # noqa: E402
from cabinet_candidate_filter_sweep import replay_global_h, map_canvas_box_to_reference, rect_overlap, write_json  # noqa: E402

STANDARD = Path(__file__).with_name("cabinet_material2_qualitative_manual_standard_v1.json")
TARGETS = {
    "ChatGPT Image 2026年8月25日 16_23_59 (1).png": {"Q01"},
    "ChatGPT Image 2026年8月25日 16_24_00 (2).png": {"Q01"},
}
FIXED_ROI = [0.50, 0.15, 0.75, 0.58]


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot read {path}")
    return image


def bright_line_candidates(reference: np.ndarray, aligned: np.ndarray, value_min: int, diff_min: int) -> tuple[np.ndarray, list[dict]]:
    height, width = reference.shape[:2]
    x1, y1 = int(width * FIXED_ROI[0]), int(height * FIXED_ROI[1])
    x2, y2 = int(width * FIXED_ROI[2]), int(height * FIXED_ROI[3])
    hsv = cv2.cvtColor(aligned, cv2.COLOR_BGR2HSV)
    difference = cv2.cvtColor(cv2.absdiff(reference, aligned), cv2.COLOR_BGR2GRAY)
    # New/changed bright, low-saturation sheath pixels.  The 3x13 close joins
    # a cable's interrupted printed/edge pixels while avoiding horizontal rails.
    mask = ((hsv[:, :, 2] >= value_min) & (hsv[:, :, 1] <= 105) & (difference >= diff_min)).astype(np.uint8) * 255
    roi_mask = np.zeros_like(mask)
    roi_mask[y1:y2, x1:x2] = 255
    mask &= roi_mask
    joined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 13)))
    joined = cv2.morphologyEx(joined, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 3)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    candidates = []
    for label in range(1, count):
        left, top, box_w, box_h, area = (int(value) for value in stats[label])
        if area < 75 or box_h < 24 or box_h < box_w * 1.4:
            continue
        candidates.append({"left": max(0, left - 5), "top": max(0, top - 5), "right": min(width, left + box_w + 5), "bottom": min(height, top + box_h + 5), "area": area})
    candidates.sort(key=lambda item: item["area"], reverse=True)
    return joined, candidates[:12]


def draw(path: Path, image: np.ndarray, groups: list[dict], candidates: list[dict], title: str) -> None:
    canvas = image.copy()
    for group in groups:
        x1, y1, x2, y2 = group["mapped_reference_bbox"]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (70, 220, 70), 2)
        cv2.putText(canvas, group["id"], (x1, max(18, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (70, 220, 70), 2, cv2.LINE_AA)
    for index, candidate in enumerate(candidates, 1):
        cv2.rectangle(canvas, (candidate["left"], candidate["top"]), (candidate["right"], candidate["bottom"]), (255, 0, 255), 2)
        cv2.putText(canvas, f"L{index}", (candidate["left"], min(canvas.shape[0] - 4, candidate["bottom"] + 17)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(canvas, title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA)
    ok, encoded = cv2.imencode(".png", canvas)
    if not ok:
        raise RuntimeError(f"Cannot encode {path}")
    encoded.tofile(str(path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("material2_thin_bright_line_probe_20260827"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    standard = json.loads(STANDARD.read_text(encoding="utf-8"))
    reference = read_image(ROOT / "2.png")
    canvas_size = tuple(standard["coordinate_space"]["canvas_size"])
    variants = [("v150_d30", 150, 30), ("v165_d40", 165, 40), ("v180_d50", 180, 50)]
    results = []
    for case in standard["images"]:
        if case["image"] not in TARGETS:
            continue
        inspection = read_image(ROOT / case["image"])
        aligned, alignment = automatic_homography(reference, inspection)
        if aligned is None:
            raise RuntimeError(f"Alignment rejected: {case['image']}")
        h = replay_global_h(reference, inspection)
        groups = [{
            "id": group["id"], "manual_canvas_bbox": group["bbox"], "description": group["description"],
            "mapped_reference_bbox": map_canvas_box_to_reference(group["bbox"], (inspection.shape[1], inspection.shape[0]), h, canvas_size=canvas_size, destination_size=(reference.shape[1], reference.shape[0])),
        } for group in case["groups"] if group["id"] in TARGETS[case["image"]]]
        outcome = {"image": case["image"], "alignment": alignment, "groups": groups, "variants": []}
        for name, value_min, diff_min in variants:
            mask, candidates = bright_line_candidates(reference, aligned, value_min, diff_min)
            cover = []
            for group in groups:
                overlaps = [rect_overlap(group["mapped_reference_bbox"], candidate) for candidate in candidates]
                cover.append({"id": group["id"], "proxy_coverage": any(bool(item["proxy_hit"]) for item in overlaps), "candidate_overlaps": overlaps})
            outcome["variants"].append({"name": name, "value_min": value_min, "diff_min": diff_min, "candidates": candidates, "coverage": cover})
            stem = Path(case["image"]).stem
            draw(args.output / f"{stem}_{name}.png", aligned, groups, candidates, f"bright thin-line {name}; green=manual, magenta=candidate")
            ok, encoded = cv2.imencode(".png", mask)
            if ok:
                encoded.tofile(str(args.output / f"{stem}_{name}_mask.png"))
        results.append(outcome)
    write_json(args.output / "summary.json", {"purpose": "Independent high-resolution thin bright-line residual probe, fixed upper-middle ROI only; no mainline change.", "cases": results})
    for case in results:
        for variant in case["variants"]:
            hits = sum(item["proxy_coverage"] for item in variant["coverage"])
            print(case["image"], variant["name"], f"candidates={len(variant['candidates'])}", f"proxy={hits}/{len(case['groups'])}")


if __name__ == "__main__":
    main()
