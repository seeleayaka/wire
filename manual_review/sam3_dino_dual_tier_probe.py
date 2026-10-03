"""Experimental dual-tier review: DINO-confirmed vs SAM-led wire evidence.

Green regions retain the existing requirement of a DINO/traditional candidate
with SAM mask support.  Yellow regions are *not* production alarms: they are
independent, unjoined SAM3 change components whose local DINO/traditional
score is recorded, so a human can examine evidence that the hard DINO box gate
would otherwise discard.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

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
    parser.add_argument("--sam-rule-report", type=Path, required=True)
    parser.add_argument("--added-mask", type=Path, required=True)
    parser.add_argument("--missing-mask", type=Path, required=True)
    parser.add_argument("--dino-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--yellow-min-pixels", type=int, default=900)
    parser.add_argument("--yellow-min-mean", type=float, default=200.0)
    parser.add_argument("--yellow-min-p90", type=float, default=0.0)
    return parser.parse_args()


def read_image(path: Path, flags: int) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, flags)
    if image is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    ok, data = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise RuntimeError(f"Cannot write image: {path}")
    data.tofile(str(path))


def jet_to_score(heat: np.ndarray) -> np.ndarray:
    """Reverse this pipeline's exact OpenCV JET color encoding."""
    values = np.arange(256, dtype=np.uint8).reshape(256, 1)
    colors = cv2.applyColorMap(values, cv2.COLORMAP_JET).reshape(256, 3)
    lut = np.zeros(1 << 24, dtype=np.uint8)
    color_codes = colors[:, 0].astype(np.uint32) | (colors[:, 1].astype(np.uint32) << 8) | (colors[:, 2].astype(np.uint32) << 16)
    lut[color_codes] = np.arange(256, dtype=np.uint8)
    codes = heat[:, :, 0].astype(np.uint32) | (heat[:, :, 1].astype(np.uint32) << 8) | (heat[:, :, 2].astype(np.uint32) << 16)
    return lut[codes]


def boxes_overlap(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> bool:
    return max(first[0], second[0]) < min(first[2], second[2]) and max(first[1], second[1]) < min(first[3], second[3])


def mask_components(mask: np.ndarray, direction: str, min_pixels: int, min_span: int) -> list[dict[str, Any]]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    records: list[dict[str, Any]] = []
    for label in range(1, count):
        left, top, width, height, pixels = (int(value) for value in stats[label])
        if pixels < min_pixels or max(width, height) < min_span:
            continue
        records.append({
            "direction": direction,
            "bbox_xyxy": [left, top, left + width, top + height],
            "mask_pixels": pixels,
            "width": width,
            "height": height,
            "mask": labels == label,
        })
    return records


def component_count(mask: np.ndarray, bounds: tuple[int, int, int, int]) -> int:
    left, top, right, bottom = bounds
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask[top:bottom, left:right].astype(np.uint8), connectivity=8)
    return sum(int(stats[index, cv2.CC_STAT_AREA]) >= 30 for index in range(1, count))


def main() -> int:
    args = parse_args()
    reference = read_image(args.reference, cv2.IMREAD_COLOR)
    aligned = read_image(args.aligned_inspection, cv2.IMREAD_COLOR)
    added = read_image(args.added_mask, cv2.IMREAD_GRAYSCALE) > 0
    missing = read_image(args.missing_mask, cv2.IMREAD_GRAYSCALE) > 0
    if reference.shape != aligned.shape or added.shape != reference.shape[:2] or missing.shape != added.shape:
        raise RuntimeError("Reference, aligned inspection, and both SAM evidence masks must share dimensions")
    sam_rule = json.loads(args.sam_rule_report.read_text(encoding="utf-8"))
    dino_report = json.loads(args.dino_report.read_text(encoding="utf-8"))

    # This invokes exactly the current DINO/traditional mainline candidate code.
    _overlay, heat, fresh_dino_candidates = dino.dino_fused_regions(reference, aligned, [[0.01, 0.02, 0.99, 0.98]])
    score = jet_to_score(heat)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_image(args.output_dir / "dino_traditional_heat.jpg", heat)

    height, width = added.shape
    green: list[dict[str, Any]] = []
    green_boxes: list[tuple[int, int, int, int]] = []
    # Keep the high-confidence tier tied to the already accepted mainline
    # candidate report.  The fresh DINO run below exists only to obtain a dense
    # score map for SAM-led audit, never to replace or widen that candidate set.
    for index, candidate in enumerate(dino_report.get("candidates", []), start=1):
        left, top = max(0, int(candidate["left"])), max(0, int(candidate["top"]))
        right, bottom = min(width, int(candidate["right"])), min(height, int(candidate["bottom"]))
        if right <= left or bottom <= top:
            continue
        support = int(added[top:bottom, left:right].sum() + missing[top:bottom, left:right].sum())
        if support < 120:
            continue
        bounds = (left, top, right, bottom)
        green_boxes.append(bounds)
        green.append({
            "id": f"green_{len(green) + 1:02d}",
            "bbox_xyxy": list(bounds),
            "sam_support_pixels": support,
            "sam_added_component_count": component_count(added, bounds),
            "sam_missing_component_count": component_count(missing, bounds),
            "dino_evidence_scores": candidate.get("evidence_scores", {}),
            "semantic": "dino_and_sam_visible_wire_related_change",
        })

    raw_components = mask_components(added, "inspection_only", args.yellow_min_pixels, 24)
    raw_components += mask_components(missing, "reference_only", args.yellow_min_pixels, 24)
    yellow: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for record in raw_components:
        mask = record.pop("mask")
        bounds = tuple(record["bbox_xyxy"])
        values = score[mask]
        p90 = float(np.percentile(values, 90)) if values.size else 0.0
        mean = float(values.mean()) if values.size else 0.0
        overlaps_green = any(boxes_overlap(bounds, green_box) for green_box in green_boxes)
        audit = {
            **record,
            "dino_traditional_score_mean": round(mean, 2),
            "dino_traditional_score_p90": round(p90, 2),
            "overlaps_green_region": overlaps_green,
        }
        audits.append(audit)
        if not overlaps_green and mean >= args.yellow_min_mean and p90 >= args.yellow_min_p90:
            yellow.append({
                "id": f"yellow_{len(yellow) + 1:02d}",
                **audit,
                "semantic": "sam_led_visible_wire_change_for_human_review",
                "not_claimed": ["confirmed_change", "physical_cable_identity", "terminal_assignment", "fault_type"],
            })

    drawing = aligned.astype(np.float32).copy()
    drawing[added] = drawing[added] * 0.62 + np.array([0, 80, 255]) * 0.38
    drawing[missing] = drawing[missing] * 0.62 + np.array([255, 160, 0]) * 0.38
    drawing = np.clip(drawing, 0, 255).astype(np.uint8)
    for item in green:
        left, top, right, bottom = item["bbox_xyxy"]
        cv2.rectangle(drawing, (left, top), (right, bottom), (0, 220, 0), 2)
        cv2.putText(drawing, item["id"], (left, max(20, top - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 0), 2, cv2.LINE_AA)
    for item in yellow:
        left, top, right, bottom = item["bbox_xyxy"]
        cv2.rectangle(drawing, (left, top), (right, bottom), (0, 215, 255), 2)
        cv2.putText(drawing, item["id"], (left, max(20, top - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 215, 255), 2, cv2.LINE_AA)
    legend = "green: DINO+SAM; yellow: SAM-led human review; red/+ added, blue/- missing"
    cv2.putText(drawing, legend, (15, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(drawing, legend, (15, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)
    write_image(args.output_dir / "dual_tier_visible_wire_changes.jpg", drawing)
    result = {
        "purpose": "Experimental dual-tier review. Yellow is SAM-led human review only, not an automatic fault verdict.",
        "inputs": {
            "sam_rule_report": str(args.sam_rule_report.resolve()),
            "dino_report_for_context": str(args.dino_report.resolve()),
            "reference": str(args.reference.resolve()),
            "aligned_inspection": str(args.aligned_inspection.resolve()),
        },
        "rules": {
        "green": "existing accepted DINO/traditional candidate report with >=120 SAM evidence pixels",
            "yellow": "independent SAM component outside green, subject to review thresholds",
            "yellow_min_pixels": args.yellow_min_pixels,
            "yellow_min_dino_traditional_score_mean": args.yellow_min_mean,
            "yellow_min_dino_traditional_score_p90": args.yellow_min_p90,
            "no_mask_joining": True,
        },
        "green_regions": green,
        "yellow_review_regions": yellow,
        "sam_component_audit": sorted(audits, key=lambda item: (item["dino_traditional_score_p90"], item["mask_pixels"]), reverse=True),
        "fresh_dino_candidate_count": len(fresh_dino_candidates),
        "prior_dino_candidate_count_for_context": len(dino_report.get("candidates", [])),
    }
    (args.output_dir / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"green": len(green), "yellow": len(yellow), "audited_sam_components": len(audits)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
