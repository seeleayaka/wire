"""Compare current cabinet alignment with a rigid-only perspective refinement.

This is an experiment, deliberately separate from the GUI/mainline.  It uses
only one extra global homography estimated from fixed cabinet surfaces; it does
not use optical flow, mesh warps, or any per-pixel deformation.  Therefore a
straight line remains straight under every applied transform.

The output is for manual review.  Candidate totals are not fault accuracy.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "prototype"))

# Import torch/DINO before the PyQt-dependent review modules on this Windows host.
from assembly_auto_review_dino_v2 import dino_fused_regions  # noqa: E402
import assembly_auto_review_robust_v3 as perspective  # noqa: E402


# These are deliberately broad, cabinet-4-specific *rigid* surfaces: cabinet
# frame/ducts and component faces.  Cable channels and free wire ends are not
# anchors.  The mask is written alongside every probe so it remains auditable.
RIGID_ANCHOR_ROIS: tuple[tuple[float, float, float, float], ...] = (
    (0.00, 0.00, 1.00, 0.09),  # upper cabinet duct
    (0.00, 0.40, 1.00, 0.50),  # middle cabinet duct
    (0.00, 0.80, 1.00, 0.90),  # lower cabinet duct
    (0.00, 0.00, 0.06, 0.90),  # left frame
    (0.94, 0.00, 1.00, 0.90),  # right frame
    (0.04, 0.17, 0.18, 0.36),  # upper-left fixed device faces
    (0.22, 0.23, 0.58, 0.36),  # upper breaker faces
    (0.18, 0.57, 0.56, 0.74),  # lower breaker faces
    (0.59, 0.58, 0.87, 0.75),  # lower fixed contactor faces
)

# A conservative measurement mask, intentionally narrower than the ECC anchor
# mask.  It contains only cabinet rails/frame/ducts that should remain static;
# the breaker and terminal faces are omitted because a real local change there
# must reduce *content* agreement, not be mistaken for registration failure.
STATIC_OVERLAP_ROIS: tuple[tuple[float, float, float, float], ...] = (
    (0.00, 0.015, 1.00, 0.085),  # upper horizontal duct
    (0.00, 0.405, 1.00, 0.495),  # middle horizontal duct
    (0.00, 0.805, 1.00, 0.895),  # lower horizontal duct
    (0.015, 0.090, 0.070, 0.900),  # left cabinet frame/duct
    (0.930, 0.090, 0.985, 0.900),  # right cabinet frame/duct
)


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix if path.suffix else ".jpg"
    okay, encoded = cv2.imencode(suffix, image)
    if not okay:
        raise ValueError(f"Cannot encode image: {path}")
    encoded.tofile(str(path))


def natural_key(path: Path) -> list[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def normalized_gray(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(cv2.normalize(gray, None, 0.0, 1.0, cv2.NORM_MINMAX).astype(np.float32), (5, 5), 0)


def rigid_anchor_mask(reference: np.ndarray) -> np.ndarray:
    height, width = reference.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    for left, top, right, bottom in RIGID_ANCHOR_ROIS:
        x1, x2 = round(left * width), round(right * width)
        y1, y2 = round(top * height), round(bottom * height)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
    # Do not let a mask edge itself become an ECC alignment signal.
    return cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))


def static_overlap_mask(reference: np.ndarray) -> np.ndarray:
    """Mask fixed cabinet structures for an alignment-only overlap measurement."""
    height, width = reference.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    for left, top, right, bottom in STATIC_OVERLAP_ROIS:
        x1, x2 = round(left * width), round(right * width)
        y1, y2 = round(top * height), round(bottom * height)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
    return cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))


def masked_overlap_metrics(
    reference: np.ndarray,
    aligned: np.ndarray,
    valid_warp: np.ndarray,
    static_mask: np.ndarray,
) -> dict[str, Any]:
    """Report static-structure overlap without treating changed parts as alignment error.

    `intensity_correlation` is brightness-normalised Pearson correlation after
    light blur.  `edge_overlap_f1` permits a two-pixel tolerance in either
    image, so it expresses whether fixed edges land on the same structures
    rather than punishing sub-pixel interpolation.
    """
    mask = (static_mask > 0) & (valid_warp > 0)
    pixel_count = int(np.count_nonzero(mask))
    if pixel_count < 500:
        return {"available": False, "reason": "insufficient_static_valid_pixels"}

    ref_gray = cv2.GaussianBlur(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    aligned_gray = cv2.GaussianBlur(cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    ref_values = ref_gray[mask].astype(np.float32)
    aligned_values = aligned_gray[mask].astype(np.float32)
    ref_std = float(ref_values.std())
    aligned_std = float(aligned_values.std())
    if ref_std < 1e-6 or aligned_std < 1e-6:
        correlation: float | None = None
    else:
        correlation = float(np.corrcoef(ref_values, aligned_values)[0, 1])

    reference_edges = cv2.Canny(ref_gray, 45, 120)
    aligned_edges = cv2.Canny(aligned_gray, 45, 120)
    reference_edges[~mask] = 0
    aligned_edges[~mask] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    reference_dilated = cv2.dilate(reference_edges, kernel)
    aligned_dilated = cv2.dilate(aligned_edges, kernel)
    ref_edge_count = int(np.count_nonzero(reference_edges))
    aligned_edge_count = int(np.count_nonzero(aligned_edges))
    if ref_edge_count == 0 or aligned_edge_count == 0:
        edge_precision: float | None = None
        edge_recall: float | None = None
        edge_f1: float | None = None
    else:
        edge_precision = float(np.count_nonzero((aligned_edges > 0) & (reference_dilated > 0)) / aligned_edge_count)
        edge_recall = float(np.count_nonzero((reference_edges > 0) & (aligned_dilated > 0)) / ref_edge_count)
        edge_f1 = float(2.0 * edge_precision * edge_recall / (edge_precision + edge_recall)) if edge_precision + edge_recall else 0.0

    return {
        "available": True,
        "metric_scope": "fixed cabinet rails_and_frames_only",
        "static_valid_pixel_coverage": round(pixel_count / static_mask.size, 4),
        "intensity_correlation": None if correlation is None else round(correlation, 4),
        "mean_abs_gray_difference_0_255": round(float(np.mean(np.abs(ref_values - aligned_values))), 3),
        "edge_tolerance_pixels": 2,
        "reference_static_edge_pixels": ref_edge_count,
        "aligned_static_edge_pixels": aligned_edge_count,
        "edge_overlap_precision": None if edge_precision is None else round(edge_precision, 4),
        "edge_overlap_recall": None if edge_recall is None else round(edge_recall, 4),
        "edge_overlap_f1": None if edge_f1 is None else round(edge_f1, 4),
    }


def _forward_matrix_from_ecc_inverse(ecc_warp: np.ndarray) -> np.ndarray:
    """ECC's recommended WARP_INVERSE_MAP form implies this forward matrix."""
    forward = np.linalg.inv(ecc_warp)
    return forward / forward[2, 2]


def rigid_homography_refine(
    reference: np.ndarray,
    globally_aligned: np.ndarray,
    globally_valid: np.ndarray,
    anchor_mask: np.ndarray,
    min_correlation: float,
    max_corner_shift: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Refine with one masked homography; never a mesh or pixel displacement field."""
    height, width = reference.shape[:2]
    ecc_warp = np.eye(3, dtype=np.float32)
    report: dict[str, Any] = {
        "method": "masked_global_ecc_homography_after_sift",
        "geometry_contract": "single_3x3_homography_only_no_mesh_no_optical_flow",
        "anchor_mask_coverage": round(float(np.count_nonzero(anchor_mask)) / anchor_mask.size, 4),
        "min_correlation_gate": min_correlation,
        "max_corner_shift_gate_pixels": max_corner_shift,
        "applied": False,
    }
    try:
        correlation, ecc_warp = cv2.findTransformECC(
            normalized_gray(reference),
            normalized_gray(globally_aligned),
            ecc_warp,
            cv2.MOTION_HOMOGRAPHY,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 120, 1e-7),
            anchor_mask,
            5,
        )
        forward = _forward_matrix_from_ecc_inverse(ecc_warp)
        corners = np.float32([[[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]])
        shifted = cv2.perspectiveTransform(corners, forward)
        max_corner_shift_pixels = float(np.linalg.norm(shifted - corners, axis=2).max())
        report.update(
            correlation=round(float(correlation), 5),
            max_corner_shift_pixels=round(max_corner_shift_pixels, 3),
        )
        # A small post-SIFT correction is allowed.  A large one is likely an
        # unstable ECC optimum rather than a capture-pose refinement.
        if correlation < min_correlation or max_corner_shift_pixels > max_corner_shift:
            report["reason"] = "ecc_refinement_rejected_by_correlation_or_corner_motion_gate"
            return globally_aligned, globally_valid, report
        refined = cv2.warpPerspective(
            globally_aligned,
            ecc_warp,
            (width, height),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
        )
        valid = cv2.warpPerspective(
            globally_valid,
            ecc_warp,
            (width, height),
            flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        valid = cv2.erode(valid, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        report.update(
            applied=True,
            valid_warp_coverage=round(float(np.count_nonzero(valid)) / valid.size, 4),
            reason="passed",
        )
        return refined, valid, report
    except cv2.error as error:
        report["reason"] = "ecc_not_converged"
        report["opencv_error"] = str(error).split("\n", maxsplit=1)[0]
        return globally_aligned, globally_valid, report


def labelled_strip(reference: np.ndarray, base: np.ndarray, refined: np.ndarray) -> np.ndarray:
    panels = [reference.copy(), base.copy(), refined.copy()]
    labels = ["reference", "SIFT + MAGSAC homography", "rigid-anchor ECC homography"]
    for panel, label in zip(panels, labels):
        cv2.rectangle(panel, (0, 0), (min(panel.shape[1] - 1, 420), 36), (25, 25, 25), thickness=-1)
        cv2.putText(panel, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return np.hstack(panels)


def overlap_blend_strip(reference: np.ndarray, base: np.ndarray, refined: np.ndarray, static_mask: np.ndarray) -> np.ndarray:
    """Manual aid: white/grey fixed structures mean the aligned images overlap."""
    panels = [reference.copy()]
    for aligned in (base, refined):
        panel = reference.copy()
        blended = cv2.addWeighted(reference, 0.5, aligned, 0.5, 0)
        panel[static_mask > 0] = blended[static_mask > 0]
        panels.append(panel)
    labels = ["reference", "SIFT overlap on fixed structures", "rigid-ECC overlap on fixed structures"]
    for panel, label in zip(panels, labels):
        cv2.rectangle(panel, (0, 0), (min(panel.shape[1] - 1, 500), 36), (25, 25, 25), thickness=-1)
        cv2.putText(panel, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return np.hstack(panels)


def run_case(
    reference: np.ndarray,
    inspection_path: Path,
    output: Path,
    roi: list[float],
    anchor_mask: np.ndarray,
    overlap_mask: np.ndarray,
    min_correlation: float,
    max_corner_shift: float,
    alignment_only: bool,
) -> dict[str, Any]:
    inspection = read_image(inspection_path)
    globally_aligned, base_report = perspective.automatic_homography(reference, inspection)
    case: dict[str, Any] = {"image": str(inspection_path), "base_alignment": base_report}
    if globally_aligned is None:
        case["decision"] = "alignment_uncertain_manual_review"
        return case
    base_valid = perspective.auto.LAST_WARP_VALID_MASK
    if not isinstance(base_valid, np.ndarray):
        raise RuntimeError("Global alignment passed without a valid-warp mask")
    refined, refined_valid, refinement = rigid_homography_refine(
        reference,
        globally_aligned,
        base_valid,
        anchor_mask,
        min_correlation,
        max_corner_shift,
    )
    case["rigid_refinement"] = refinement

    case_dir = output / inspection_path.stem
    write_image(case_dir / "alignment_strip.jpg", labelled_strip(reference, globally_aligned, refined))
    write_image(case_dir / "static_overlap_blend_strip.jpg", overlap_blend_strip(reference, globally_aligned, refined, overlap_mask))
    write_image(case_dir / "base_aligned.jpg", globally_aligned)
    write_image(case_dir / "rigid_refined_aligned.jpg", refined)
    case["base_static_overlap"] = masked_overlap_metrics(reference, globally_aligned, base_valid, overlap_mask)
    case["rigid_static_overlap"] = masked_overlap_metrics(reference, refined, refined_valid, overlap_mask)

    if alignment_only:
        case["decision"] = "alignment_overlap_manual_review"
        case["candidate_note"] = "Skipped in alignment-only mode. No candidate counts were produced."
        return case

    perspective.auto.LAST_WARP_VALID_MASK = base_valid
    base_overlay, _base_heat, base_candidates = dino_fused_regions(reference, globally_aligned, [roi])
    perspective.auto.LAST_WARP_VALID_MASK = refined_valid
    refined_overlay, _refined_heat, refined_candidates = dino_fused_regions(reference, refined, [roi])
    write_image(case_dir / "base_candidates.jpg", base_overlay)
    write_image(case_dir / "rigid_refined_candidates.jpg", refined_overlay)
    case.update(
        decision="possible_difference_manual_review",
        base_candidate_count=len(base_candidates),
        rigid_refined_candidate_count=len(refined_candidates),
        candidate_note="Counts are diagnostic candidate volume only, not fault accuracy.",
    )
    return case


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--names", nargs="*", help="Optional filenames; omit to probe every supported image in input-dir.")
    parser.add_argument("--roi", default="0,0,1,1", help="Normalized l,t,r,b review ROI; default is full cabinet.")
    parser.add_argument("--ecc-min-correlation", type=float, default=0.72, help="Masked ECC acceptance floor; default 0.72.")
    parser.add_argument("--ecc-max-corner-shift", type=float, default=24.0, help="Maximum global corner displacement in pixels; default 24.")
    parser.add_argument("--alignment-only", action="store_true", help="Skip DINO/candidate generation; report alignment overlap only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roi = [float(value) for value in args.roi.split(",")]
    if len(roi) != 4 or not (0.0 <= roi[0] < roi[2] <= 1.0 and 0.0 <= roi[1] < roi[3] <= 1.0):
        raise SystemExit("--roi must be four normalized values l,t,r,b")
    reference = read_image(args.reference)
    args.output.mkdir(parents=True, exist_ok=True)
    anchor_mask = rigid_anchor_mask(reference)
    write_image(args.output / "rigid_anchor_mask.png", anchor_mask)
    overlap_mask = static_overlap_mask(reference)
    write_image(args.output / "static_overlap_mask.png", overlap_mask)
    if args.names:
        inputs = [args.input_dir / name for name in args.names]
    else:
        inputs = sorted((path for path in args.input_dir.iterdir() if path.suffix.lower() in {".bmp", ".jpeg", ".jpg", ".png", ".webp"} and path != args.reference), key=natural_key)
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing input images: {missing}")
    cases = []
    for index, image_path in enumerate(inputs, 1):
        case = run_case(
            reference,
            image_path,
            args.output,
            roi,
            anchor_mask,
            overlap_mask,
            args.ecc_min_correlation,
            args.ecc_max_corner_shift,
            args.alignment_only,
        )
        case["index"] = index
        cases.append(case)
        print(json.dumps({"index": index, "image": image_path.name, "base_edge_f1": case.get("base_static_overlap", {}).get("edge_overlap_f1"), "rigid_edge_f1": case.get("rigid_static_overlap", {}).get("edge_overlap_f1"), "ecc": case.get("rigid_refinement", {}).get("reason")}, ensure_ascii=False), flush=True)
    summary = {
        "kind": "rigid_homography_alignment_probe",
        "contract": "single_3x3_homography_only; no mesh, no optical flow, no generated pixels",
        "reference": str(args.reference),
        "rigid_anchor_rois_normalized": RIGID_ANCHOR_ROIS,
        "static_overlap_rois_normalized": STATIC_OVERLAP_ROIS,
        "roi": roi,
        "ecc_gates": {"min_correlation": args.ecc_min_correlation, "max_corner_shift_pixels": args.ecc_max_corner_shift},
        "cases": cases,
        "metric_boundary": "Static overlap measures only registration of cabinet rails/frame. It is not fault truth, precision, recall, or field accuracy.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
