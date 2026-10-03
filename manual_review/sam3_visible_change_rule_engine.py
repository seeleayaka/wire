"""Detect visible wire-related changes from already generated SAM3 masks.

This independent review prototype never reruns SAM3, modifies masks, joins
separate masks into a physical cable, assigns terminals, or names a fault.
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
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--inspection-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--proximity-px", type=int, default=8)
    parser.add_argument("--min-component-pixels", type=int, default=100)
    parser.add_argument("--min-component-span-px", type=int, default=24)
    parser.add_argument("--manual-standard", type=Path)
    parser.add_argument(
        "--already-aligned",
        action="store_true",
        help="Compare masks directly because SAM3 was run after the mainline image alignment.",
    )
    parser.add_argument(
        "--valid-mask",
        type=Path,
        help="Optional valid pixels from the alignment step; excludes warp padding from comparison.",
    )
    parser.add_argument(
        "--manual-image-name",
        help="Original inspection filename used to find an entry in --manual-standard.",
    )
    return parser.parse_args()


def read_image(path: Path, flags: int) -> np.ndarray:
    raw = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(raw, flags)
    if image is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise RuntimeError(f"Cannot encode image: {path}")
    encoded.tofile(str(path))


def source_paths(directory: Path) -> tuple[Path, Path, Path]:
    paths = (directory / "input.jpg", directory / "mask_union.png", directory / "report.json")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("SAM3 source directory is incomplete: " + ", ".join(map(str, missing)))
    return paths


def estimate_homography(reference: np.ndarray, inspection: np.ndarray) -> tuple[np.ndarray, dict[str, float | int]]:
    """Use the existing project SIFT/MAGSAC settings for inspection-to-reference H."""
    sift = cv2.SIFT_create(nfeatures=9000, contrastThreshold=0.014, edgeThreshold=12)
    ref_kp, ref_desc = sift.detectAndCompute(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), None)
    ins_kp, ins_desc = sift.detectAndCompute(cv2.cvtColor(inspection, cv2.COLOR_BGR2GRAY), None)
    if ref_desc is None or ins_desc is None:
        raise RuntimeError("Cannot estimate H: no SIFT descriptors")
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(ins_desc, ref_desc, k=2)
    good = [first for first, second in pairs if first.distance < 0.70 * second.distance]
    if len(good) < 60:
        raise RuntimeError(f"Cannot estimate H: only {len(good)} ratio-test matches")
    source = np.float32([ins_kp[item.queryIdx].pt for item in good]).reshape(-1, 1, 2)
    destination = np.float32([ref_kp[item.trainIdx].pt for item in good]).reshape(-1, 1, 2)
    h, inlier_mask = cv2.findHomography(
        source,
        destination,
        method=getattr(cv2, "USAC_MAGSAC", cv2.RANSAC),
        ransacReprojThreshold=4.0,
        maxIters=10000,
        confidence=0.995,
    )
    if h is None or inlier_mask is None:
        raise RuntimeError("Cannot estimate H: MAGSAC did not produce a homography")
    inliers = inlier_mask.ravel().astype(bool)
    projected = cv2.perspectiveTransform(source[inliers].reshape(-1, 1, 2), h).reshape(-1, 2)
    errors = np.linalg.norm(projected - destination[inliers].reshape(-1, 2), axis=1)
    return h, {
        "ratio_test_matches": len(good),
        "inliers": int(inliers.sum()),
        "median_reprojection_error_px": round(float(np.median(errors)), 4),
    }


def dilate(binary: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return binary.astype(bool)
    size = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate(binary.astype(np.uint8), kernel).astype(bool)


def extract_components(
    binary: np.ndarray, direction: str, min_pixels: int, min_span: int
) -> tuple[list[dict[str, int | str]], np.ndarray]:
    """Keep separate local change components. This deliberately does not join masks."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    kept = np.zeros(binary.shape, dtype=bool)
    candidates: list[dict[str, int | str]] = []
    for label in range(1, count):
        left = int(stats[label, cv2.CC_STAT_LEFT])
        top = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_pixels or max(width, height) < min_span:
            continue
        kept |= labels == label
        candidates.append(
            {
                "direction": direction,
                "left": left,
                "top": top,
                "right": left + width,
                "bottom": top + height,
                "component_pixels": area,
                "width": width,
                "height": height,
            }
        )
    candidates.sort(key=lambda item: int(item["component_pixels"]), reverse=True)
    return candidates, kept


def render_overlay(
    inspection_aligned: np.ndarray,
    inspection_only: np.ndarray,
    reference_only: np.ndarray,
    candidates: list[dict[str, int | str]],
) -> np.ndarray:
    drawing = inspection_aligned.astype(np.float32).copy()
    drawing[inspection_only] = drawing[inspection_only] * 0.35 + np.array([0, 80, 255]) * 0.65
    drawing[reference_only] = drawing[reference_only] * 0.35 + np.array([255, 160, 0]) * 0.65
    drawing = np.clip(drawing, 0, 255).astype(np.uint8)
    for index, candidate in enumerate(candidates, start=1):
        added = candidate["direction"] == "inspection_only"
        color = (0, 0, 255) if added else (255, 160, 0)
        left, top = int(candidate["left"]), int(candidate["top"])
        right, bottom = int(candidate["right"]), int(candidate["bottom"])
        cv2.rectangle(drawing, (left, top), (right, bottom), color, 2)
        cv2.putText(drawing, f"C{index} {'+' if added else '-'}", (left, max(18, top - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)
    legend = "red/+ inspection-only mask; blue/- reference-only mask"
    cv2.putText(drawing, legend, (15, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(drawing, legend, (15, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (25, 25, 25), 1, cv2.LINE_AA)
    return drawing


def manual_overlap_report(
    standard_path: Path,
    inspection_report: dict[str, Any],
    candidates: list[dict[str, int | str]],
    manual_image_name: str | None,
) -> dict[str, Any] | None:
    standard = json.loads(standard_path.read_text(encoding="utf-8"))
    name = manual_image_name or Path(str(inspection_report["input"])).name
    annotation = next((item for item in standard["images"] if item["image"] == name), None)
    if annotation is None or not annotation.get("groups"):
        return None
    groups = []
    for group in annotation["groups"]:
        left, top, right, bottom = (int(value) for value in group["bbox"])
        hits = []
        for index, candidate in enumerate(candidates, start=1):
            cx1, cy1 = int(candidate["left"]), int(candidate["top"])
            cx2, cy2 = int(candidate["right"]), int(candidate["bottom"])
            intersection = max(0, min(right, cx2) - max(left, cx1)) * max(0, min(bottom, cy2) - max(top, cy1))
            area = max(1, (right - left) * (bottom - top))
            center_inside = left <= (cx1 + cx2) / 2 <= right and top <= (cy1 + cy2) / 2 <= bottom
            if center_inside or intersection / area >= 0.05:
                hits.append(index)
        groups.append(
            {
                "manual_group_id": group["id"],
                "manual_bbox_reference_xyxy": group["bbox"],
                "description": group["description"],
                "candidate_indices_proxy_hit": hits,
                "proxy_hit": bool(hits),
            }
        )
    return {
        "standard": str(standard_path.resolve()),
        "eligibility": annotation["classification"],
        "note": "Candidate-box center in manual group OR candidate box covers at least 5% of it. Review proxy only, not accuracy.",
        "groups": groups,
    }


def main() -> int:
    args = parse_args()
    ref_input, ref_union_path, ref_report_path = source_paths(args.reference_dir)
    ins_input, ins_union_path, ins_report_path = source_paths(args.inspection_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference = read_image(ref_input, cv2.IMREAD_COLOR)
    inspection = read_image(ins_input, cv2.IMREAD_COLOR)
    reference_union = read_image(ref_union_path, cv2.IMREAD_GRAYSCALE) > 0
    inspection_union = read_image(ins_union_path, cv2.IMREAD_GRAYSCALE) > 0
    reference_report = json.loads(ref_report_path.read_text(encoding="utf-8"))
    inspection_report = json.loads(ins_report_path.read_text(encoding="utf-8"))
    if reference_union.shape != reference.shape[:2] or inspection_union.shape != inspection.shape[:2]:
        raise RuntimeError("SAM3 union mask and input image dimensions do not match")
    if args.already_aligned:
        if inspection.shape[:2] != reference.shape[:2]:
            raise RuntimeError("--already-aligned requires equal reference and inspection dimensions")
        inspection_aligned = inspection
        inspection_union_aligned = inspection_union
        source_coverage = np.ones(reference.shape[:2], dtype=bool)
        alignment: dict[str, Any] = {
            "method": "prealigned_image_direct_mask_comparison",
            "reliable": True,
            "note": "The alignment acceptance report is stored beside the SAM3 input image.",
        }
    else:
        h, alignment = estimate_homography(reference, inspection)
        width, height = reference.shape[1], reference.shape[0]
        inspection_aligned = cv2.warpPerspective(inspection, h, (width, height), flags=cv2.INTER_LINEAR)
        inspection_union_aligned = cv2.warpPerspective(
            inspection_union.astype(np.uint8), h, (width, height), flags=cv2.INTER_NEAREST
        ).astype(bool)
        source_coverage = cv2.warpPerspective(
            np.full(inspection_union.shape, 255, dtype=np.uint8), h, (width, height), flags=cv2.INTER_NEAREST
        ).astype(bool)
    if args.valid_mask:
        valid_mask = read_image(args.valid_mask, cv2.IMREAD_GRAYSCALE) > 0
        if valid_mask.shape != reference.shape[:2]:
            raise RuntimeError("--valid-mask dimensions must match the reference image")
        source_coverage &= valid_mask
    ref_near = dilate(reference_union, args.proximity_px)
    ins_near = dilate(inspection_union_aligned, args.proximity_px)
    inspection_only_raw = inspection_union_aligned & source_coverage & ~ref_near
    reference_only_raw = reference_union & source_coverage & ~ins_near
    inspection_candidates, inspection_only = extract_components(
        inspection_only_raw, "inspection_only", args.min_component_pixels, args.min_component_span_px
    )
    reference_candidates, reference_only = extract_components(
        reference_only_raw, "reference_only", args.min_component_pixels, args.min_component_span_px
    )
    candidates = sorted(inspection_candidates + reference_candidates, key=lambda item: int(item["component_pixels"]), reverse=True)
    write_image(args.output_dir / "inspection_aligned.jpg", inspection_aligned)
    write_image(args.output_dir / "inspection_only_mask.png", inspection_only.astype(np.uint8) * 255)
    write_image(args.output_dir / "reference_only_mask.png", reference_only.astype(np.uint8) * 255)
    write_image(args.output_dir / "visible_wire_change_candidates.jpg", render_overlay(inspection_aligned, inspection_only, reference_only, candidates))
    result: dict[str, Any] = {
        "purpose": "Visible wire-related change candidates from existing independent SAM3 masks; no cable reconstruction or fault classification.",
        "reference_sam3": {"directory": str(args.reference_dir.resolve()), "input": str(ref_input.resolve()), "prompt": reference_report.get("prompt"), "threshold": reference_report.get("confidence_threshold")},
        "inspection_sam3": {"directory": str(args.inspection_dir.resolve()), "input": str(ins_input.resolve()), "prompt": inspection_report.get("prompt"), "threshold": inspection_report.get("confidence_threshold")},
        "alignment": alignment,
        "rules": {"proximity_px": args.proximity_px, "min_component_pixels": args.min_component_pixels, "min_component_span_px": args.min_component_span_px, "already_aligned": args.already_aligned, "valid_mask": str(args.valid_mask.resolve()) if args.valid_mask else None, "no_mask_joining": True},
        "candidate_count": len(candidates),
        "candidate_counts_by_direction": {"inspection_only": len(inspection_candidates), "reference_only": len(reference_candidates)},
        "candidates": candidates,
    }
    if args.manual_standard:
        result["manual_overlap_review"] = manual_overlap_report(args.manual_standard, inspection_report, candidates, args.manual_image_name)
    (args.output_dir / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"candidate_count": len(candidates), "alignment": alignment}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
