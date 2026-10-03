"""Experiment: refine a global cabinet alignment from long horizontal rails.

This probe is intentionally separate from the mainline and never creates a
mesh, optical flow, or per-pixel deformation.  After the current SIFT/MAGSAC
global homography it may apply one restricted affine matrix::

    x' = x
    y' = a*x + b*y + c

Therefore every straight line remains a straight line.  The correction only
serves to see whether the prominent horizontal cabinet rails are a useful,
auditable post-alignment constraint for a larger camera-pose change.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "prototype"))
import assembly_auto_review_robust_v3 as perspective  # noqa: E402


@dataclass(frozen=True)
class Rail:
    """A long, near-horizontal Hough segment represented as y = slope*x + intercept."""

    y: float
    slope: float
    intercept: float
    length: float


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    okay, encoded = cv2.imencode(path.suffix or ".png", image)
    if not okay:
        raise ValueError(f"Cannot encode image: {path}")
    encoded.tofile(str(path))


def detect_horizontal_rails(image: np.ndarray) -> list[Rail]:
    """Find wide horizontal structural edges, excluding image borders.

    Long cabinet ducts/rails span most of the frame, whereas component faces and
    wire labels are shorter.  The detector deliberately returns candidates for
    later matching; it does not assume every long line is a cabinet rail.
    """
    height, width = image.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.Canny(gray, 45, 135)
    edges[: round(height * 0.045), :] = 0
    edges[round(height * 0.93) :, :] = 0
    edges[:, : round(width * 0.035)] = 0
    edges[:, round(width * 0.965) :] = 0
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 720,
        threshold=max(70, round(width * 0.12)),
        minLineLength=round(width * 0.42),
        maxLineGap=round(width * 0.045),
    )
    raw: list[Rail] = []
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            dx = float(x2 - x1)
            dy = float(y2 - y1)
            length = float(np.hypot(dx, dy))
            if length < width * 0.42 or abs(dx) < 1.0:
                continue
            slope = dy / dx
            if abs(slope) > 0.055:  # approximately 3.1 degrees
                continue
            intercept = float(y1 - slope * x1)
            raw.append(Rail(y=float(slope * width / 2.0 + intercept), slope=slope, intercept=intercept, length=length))

    # Hough commonly returns duplicate fragments on the two sharp edges of one
    # rail.  Cluster only near-identical y values, preserving distinct upper and
    # lower rail edges that are physically separated.
    clusters: list[list[Rail]] = []
    for rail in sorted(raw, key=lambda item: item.y):
        if not clusters or abs(rail.y - clusters[-1][-1].y) > max(5.0, height * 0.010):
            clusters.append([rail])
        else:
            clusters[-1].append(rail)
    result: list[Rail] = []
    for members in clusters:
        weights = np.array([item.length for item in members], dtype=np.float64)
        result.append(
            Rail(
                y=float(np.average([item.y for item in members], weights=weights)),
                slope=float(np.average([item.slope for item in members], weights=weights)),
                intercept=float(np.average([item.intercept for item in members], weights=weights)),
                length=float(weights.sum()),
            )
        )
    return sorted(result, key=lambda item: item.length, reverse=True)[:12]


def match_rails(reference: list[Rail], aligned: list[Rail], height: int) -> list[tuple[Rail, Rail]]:
    """Match only close post-homography long-line candidates, one-to-one."""
    candidates: list[tuple[float, Rail, Rail]] = []
    limit = max(18.0, height * 0.085)
    for ref in reference:
        for test in aligned:
            distance = abs(ref.y - test.y)
            if distance <= limit:
                # Prefer long edges and compatible orientation, but let the
                # y-residual dominate because global H has already been applied.
                score = distance + height * 0.55 * abs(ref.slope - test.slope) - 0.0002 * min(ref.length, test.length)
                candidates.append((score, ref, test))
    pairs: list[tuple[Rail, Rail]] = []
    used_reference: set[Rail] = set()
    used_aligned: set[Rail] = set()
    for _, ref, test in sorted(candidates, key=lambda item: item[0]):
        if ref in used_reference or test in used_aligned:
            continue
        pairs.append((ref, test))
        used_reference.add(ref)
        used_aligned.add(test)
    return sorted(pairs, key=lambda pair: pair[0].y)


def fit_y_axis_affine(pairs: list[tuple[Rail, Rail]], height: int) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Fit a constrained affine y correction from corresponding horizontal lines."""
    report: dict[str, Any] = {"method": "rail_constrained_y_axis_affine", "pair_count": len(pairs), "applied": False}
    if len(pairs) < 2:
        report["reason"] = "fewer_than_two_matched_long_horizontal_edges"
        return None, report
    rows: list[list[float]] = []
    values: list[float] = []
    for reference, aligned in pairs:
        # y' = a*x + b*y + c, and y=m*x+k for each rail.
        rows.append([1.0, aligned.slope, 0.0])
        values.append(reference.slope)
        rows.append([0.0, aligned.intercept, 1.0])
        values.append(reference.intercept)
    coefficients, *_ = np.linalg.lstsq(np.asarray(rows), np.asarray(values), rcond=None)
    shear, vertical_scale, vertical_offset = (float(value) for value in coefficients)
    matrix = np.array([[1.0, 0.0, 0.0], [shear, vertical_scale, vertical_offset], [0.0, 0.0, 1.0]], dtype=np.float32)

    def residual(transform: np.ndarray | None) -> float:
        errors: list[float] = []
        for reference, aligned in pairs:
            for x in (0.0, 1.0):
                before_y = aligned.slope * x + aligned.intercept
                expected_y = reference.slope * x + reference.intercept
                if transform is not None:
                    vector = transform @ np.array([x, before_y, 1.0], dtype=np.float32)
                    before_y = float(vector[1] / vector[2])
                errors.append(abs(before_y - expected_y))
        return float(np.mean(errors))

    before = residual(None)
    after = residual(matrix)
    report.update(
        shear=round(shear, 6),
        vertical_scale=round(vertical_scale, 6),
        vertical_offset_pixels=round(vertical_offset, 3),
        rail_mean_residual_before_pixels=round(before, 3),
        rail_mean_residual_after_pixels=round(after, 3),
    )
    safe_geometry = abs(shear) <= 0.05 and 0.90 <= vertical_scale <= 1.10 and abs(vertical_offset) <= height * 0.09
    helpful = after + 1.0 < before and after <= before * 0.70
    if not safe_geometry:
        report["reason"] = "correction_exceeds_straight_line_safe_geometry_gate"
        return None, report
    if not helpful:
        report["reason"] = "matched_rail_residual_not_meaningfully_improved"
        return None, report
    report.update(applied=True, reason="passed")
    return matrix, report


def blend(reference: np.ndarray, other: np.ndarray) -> np.ndarray:
    return cv2.addWeighted(reference, 0.5, other, 0.5, 0)


def draw_rails(image: np.ndarray, rails: list[Rail], color: tuple[int, int, int], label: str) -> np.ndarray:
    result = image.copy()
    height, width = result.shape[:2]
    for index, rail in enumerate(rails):
        start = (0, round(rail.intercept))
        end = (width - 1, round(rail.slope * (width - 1) + rail.intercept))
        cv2.line(result, start, end, color, 2, cv2.LINE_AA)
        cv2.putText(result, str(index + 1), (12, max(20, start[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    cv2.rectangle(result, (0, 0), (min(width - 1, 560), 34), (25, 25, 25), thickness=-1)
    cv2.putText(result, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return result


def run(reference_path: Path, inspection_path: Path, output: Path) -> dict[str, Any]:
    reference = read_image(reference_path)
    inspection = read_image(inspection_path)
    base, base_report = perspective.automatic_homography(reference, inspection)
    output.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "kind": "rail_constrained_straight_line_alignment_probe",
        "geometry_contract": "global SIFT/MAGSAC H followed only by x'=x, y'=a*x+b*y+c; no mesh, optical flow, or per-pixel deformation",
        "reference": str(reference_path),
        "inspection": str(inspection_path),
        "base_alignment": base_report,
    }
    if base is None:
        report["decision"] = "alignment_uncertain_manual_review"
        (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report

    reference_rails = detect_horizontal_rails(reference)
    aligned_rails = detect_horizontal_rails(base)
    pairs = match_rails(reference_rails, aligned_rails, reference.shape[0])
    transform, rail_report = fit_y_axis_affine(pairs, reference.shape[0])
    report["reference_rails"] = [asdict(item) for item in reference_rails]
    report["base_aligned_rails"] = [asdict(item) for item in aligned_rails]
    report["matched_rails"] = [{"reference": asdict(ref), "base_aligned": asdict(test)} for ref, test in pairs]
    report["rail_refinement"] = rail_report

    write_image(output / "reference_with_detected_rails.jpg", draw_rails(reference, reference_rails, (0, 220, 0), "reference: detected long horizontal rails"))
    write_image(output / "base_with_detected_rails.jpg", draw_rails(base, aligned_rails, (0, 0, 230), "base alignment: detected long horizontal rails"))
    if transform is None:
        refined = base
        report["decision"] = "retain_base_alignment_manual_review"
    else:
        refined = cv2.warpPerspective(base, transform, (reference.shape[1], reference.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        report["decision"] = "rail_refined_alignment_manual_review"
    write_image(output / "base_aligned.jpg", base)
    write_image(output / "rail_refined_aligned.jpg", refined)
    panels = [reference.copy(), blend(reference, base), blend(reference, refined)]
    labels = ["reference", "SIFT/MAGSAC: 50/50 blend", "rail-constrained: 50/50 blend"]
    for panel, label in zip(panels, labels):
        cv2.rectangle(panel, (0, 0), (min(panel.shape[1] - 1, 480), 36), (25, 25, 25), thickness=-1)
        cv2.putText(panel, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
    write_image(output / "comparison_strip.jpg", np.hstack(panels))
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--inspection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run(args.reference, args.inspection, args.output)
    print(json.dumps({"decision": report["decision"], "rail_refinement": report.get("rail_refinement")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
