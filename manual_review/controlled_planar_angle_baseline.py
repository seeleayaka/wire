"""Controlled no-fault planar-angle regression for the cabinet review pipeline.

This intentionally does *not* make a realistic new camera view.  It applies a
known small 3x3 homography to one reference image, so the image content is
unchanged apart from interpolation and a bounded planar perspective change.
The existing global SIFT/MAGSAC + DINO/traditional review pipeline is then run
read-only.  Any output box is a synthetic no-fault candidate to inspect, not a
field false-positive rate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

# On this Windows host Torch must initialise before a PyQt-dependent alignment
# module is imported, otherwise c10.dll can fail with WinError 1114.
import torch  # noqa: F401
import cv2
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "prototype"))

# Import V3 first: it registers the current global SIFT/MAGSAC homography on
# the shared review object.  The DINO module then reuses that exact alignment.
import assembly_auto_review_robust_v3 as perspective  # noqa: E402
import assembly_auto_review_dino as dino  # noqa: E402
import assembly_auto_review_robust as robust  # noqa: E402


DEFAULT_REFERENCE = Path(r"C:\Users\HUAWEI\Desktop\案例\线材柜子\5\5.png")
DEFAULT_OUTPUT = Path(r"E:\PythonProject10\manual_review\controlled_planar_angle_baseline_20260827")
# This is the current cabinet recipe's full visible-cabinet check area.
CHECK_ROIS: list[list[float]] = [[0.01, 0.02, 0.99, 0.98]]

# Destination quadrilaterals in normalised reference coordinates.  The largest
# displacement is under 2.5% of a side: deliberately a *small* planar-angle
# regression, not a claim about a deep 3-D camera-pose change.
CASES: tuple[tuple[str, tuple[tuple[float, float], ...]], ...] = (
    ("identity_no_change", ((0.000, 0.000), (1.000, 0.000), (1.000, 1.000), (0.000, 1.000))),
    ("yaw_left_small", ((0.018, 0.012), (0.994, 0.003), (0.989, 0.990), (0.012, 0.997))),
    ("yaw_right_small", ((0.006, 0.003), (0.982, 0.012), (0.988, 0.997), (0.011, 0.990))),
    ("pitch_down_small", ((0.010, 0.022), (0.990, 0.016), (0.998, 0.988), (0.003, 0.994))),
    ("combined_small", ((0.022, 0.018), (0.984, 0.002), (0.995, 0.983), (0.006, 0.998))),
)


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图片：{path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    okay, encoded = cv2.imencode(path.suffix or ".png", image)
    if not okay:
        raise ValueError(f"无法编码图片：{path}")
    encoded.tofile(str(path))


def pixel_quad(width: int, height: int, normalized: tuple[tuple[float, float], ...]) -> np.ndarray:
    return np.float32([[x * (width - 1), y * (height - 1)] for x, y in normalized])


def make_planar_inspection(reference: np.ndarray, destination_normalized: tuple[tuple[float, float], ...]) -> tuple[np.ndarray, np.ndarray]:
    """Return an unchanged-content synthetic inspection image and ref->test H."""
    height, width = reference.shape[:2]
    corners = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    forward = cv2.getPerspectiveTransform(corners, pixel_quad(width, height, destination_normalized))
    inspection = cv2.warpPerspective(
        reference,
        forward,
        (width, height),
        flags=cv2.INTER_LINEAR,
        # Reflection prevents an artificial black frame from becoming an
        # apparent assembly change.  The recorded transform remains known.
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return inspection, forward


def json_safe_diagnostics() -> list[dict[str, Any]]:
    """The review diagnostics are scalar/list data, but guard JSON output anyway."""
    return json.loads(json.dumps(robust.LAST_DIAGNOSTICS, ensure_ascii=False, default=str))


def global_h_only_adaptive_regions(reference: np.ndarray, aligned: np.ndarray, rois: list[list[float]]):
    """Ablation only: keep DINO's input at global H, never tile-local ECC."""
    merged = dino.adaptive._merge_rois(rois)
    robust.LAST_DIAGNOSTICS = [
        {
            "method": "ablation_global_homography_only",
            "accepted": False,
            "accepted_coverage": 0.0,
            "comparison_mode": "global_homography_only_ablation",
        }
        for _ in merged
    ]
    robust.LAST_LOCAL_ALIGNED = None
    robust.LAST_LOCAL_ACCEPTED_MASK = None
    robust.LAST_LOCAL_ALIGNMENT_PARTS = []
    return aligned.copy(), np.zeros_like(reference), []


def reviewed_regions(reference: np.ndarray, aligned: np.ndarray, *, disable_local_ecc: bool):
    if not disable_local_ecc:
        return dino.dino_fused_regions(reference, aligned, CHECK_ROIS)
    # Patch only inside this experiment process.  No production/mainline module
    # is edited and the normal branch remains the default.
    with patch.object(dino.adaptive, "adaptive_regions", side_effect=global_h_only_adaptive_regions):
        return dino.dino_fused_regions(reference, aligned, CHECK_ROIS)


def run(reference_path: Path, output: Path, *, disable_local_ecc: bool = False) -> dict[str, Any]:
    reference = read_image(reference_path)
    output.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "kind": "controlled_no_fault_planar_angle_regression",
        "scope": "Known small 2-D homographies only; no fault insertion and no AI regeneration.",
        "not_a_claim": "Candidate boxes are synthetic no-fault review candidates, not field false-positive rate or production accuracy.",
        "reference": str(reference_path),
        "check_rois": CHECK_ROIS,
        "dino_alignment_input": "global_homography_only_ablation" if disable_local_ecc else "normal_mainline_local_ecc_after_global_h",
        "cases": [],
    }
    for name, destination in CASES:
        case_dir = output / name
        case_dir.mkdir(parents=True, exist_ok=True)
        inspection, forward = make_planar_inspection(reference, destination)
        write_image(case_dir / "synthetic_inspection.png", inspection)
        aligned, alignment = perspective.automatic_homography(reference, inspection)
        result: dict[str, Any] = {
            "name": name,
            "known_reference_to_inspection_h": np.round(forward, 7).tolist(),
            "destination_quad_normalized": destination,
            "alignment": alignment,
        }
        if aligned is None:
            result.update(decision="alignment_uncertain_manual_review", possible_difference_regions=None)
        else:
            overlay, heat, regions = reviewed_regions(reference, aligned, disable_local_ecc=disable_local_ecc)
            write_image(case_dir / "aligned.png", aligned)
            write_image(case_dir / "anomaly_boxes.png", overlay)
            write_image(case_dir / "heatmap.png", heat)
            result.update(
                decision="possible_difference_manual_review" if regions else "no_significant_difference",
                possible_difference_regions=len(regions),
                review_regions=regions,
                local_alignment=json_safe_diagnostics(),
            )
        (case_dir / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summary["cases"].append(result)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--disable-local-ecc", action="store_true", help="Ablation: compare candidates with global H only, without the internal tile-local ECC input.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(args.reference, args.output, disable_local_ecc=args.disable_local_ecc)
    compact = [
        {
            "name": item["name"],
            "decision": item["decision"],
            "possible_difference_regions": item["possible_difference_regions"],
        }
        for item in summary["cases"]
    ]
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
