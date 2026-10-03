"""Independent replay of manual cabinet boxes against candidate filters.

This is deliberately outside the main inspection chain.  It does not alter
candidate generation: it replays the existing chain, then applies a small
post-filter grid to the returned candidates.  The four ``local_isolated``
manual groups are only a compact regression set, not production accuracy.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Must occur before importing the PyQt-owning review modules on this Windows
# machine; otherwise torch/c10 can fail to initialize in headless replay.
import torch  # noqa: F401
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "prototype"))
import assembly_auto_review_robust_v3 as perspective  # noqa: E402
import assembly_auto_review_dino as dino  # noqa: E402

DESKTOP_CASE = Path(r"C:\Users\HUAWEI\Desktop\案例\线材柜子\4")
STANDARD = ROOT / "manual_review" / "cabinet_manual_visual_standard_v2.json"
CONTROL_SUMMARY = ROOT / "manual_review" / "controlled_planar_angle_baseline_20260827" / "summary.json"


def read_image(path: Path) -> np.ndarray:
    array = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return image


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def replay_global_h(reference: np.ndarray, inspection: np.ndarray) -> np.ndarray:
    """Repeat only the H estimation from v3 so manual canvas boxes can map.

    ``automatic_homography`` remains the authority for whether H is accepted
    and for the aligned pixels fed to DINO.  This replay is recorded and
    checked below, rather than exposing a new mainline interface.
    """
    sift = cv2.SIFT_create(nfeatures=9000, contrastThreshold=0.014, edgeThreshold=12)
    ref_kp, ref_desc = sift.detectAndCompute(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), None)
    ins_kp, ins_desc = sift.detectAndCompute(cv2.cvtColor(inspection, cv2.COLOR_BGR2GRAY), None)
    if ref_desc is None or ins_desc is None:
        raise RuntimeError("No SIFT descriptors for manual-box transform replay")
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(ins_desc, ref_desc, k=2)
    good = [first for first, second in pairs if first.distance < 0.70 * second.distance]
    if len(good) < 60:
        raise RuntimeError(f"Too few SIFT matches for manual-box transform replay: {len(good)}")
    source = np.float32([ins_kp[item.queryIdx].pt for item in good]).reshape(-1, 1, 2)
    destination = np.float32([ref_kp[item.trainIdx].pt for item in good]).reshape(-1, 1, 2)
    h, mask = cv2.findHomography(
        source, destination, method=getattr(cv2, "USAC_MAGSAC", cv2.RANSAC),
        ransacReprojThreshold=4.0, maxIters=10000, confidence=0.995,
    )
    if h is None or mask is None:
        raise RuntimeError("Manual-box transform replay did not find H")
    return h


def map_canvas_box_to_reference(
    box: list[int], raw_size: tuple[int, int], h: np.ndarray,
    canvas_size: tuple[int, int] = (1080, 712), destination_size: tuple[int, int] = (1080, 712),
) -> list[int]:
    """Map a manual canvas box through raw inspection -> H reference pixels."""
    raw_w, raw_h = raw_size
    canvas_w, canvas_h = canvas_size
    destination_w, destination_h = destination_size
    left, top, right, bottom = box
    points = np.float32([[
        [left * raw_w / canvas_w, top * raw_h / canvas_h],
        [right * raw_w / canvas_w, top * raw_h / canvas_h],
        [right * raw_w / canvas_w, bottom * raw_h / canvas_h],
        [left * raw_w / canvas_w, bottom * raw_h / canvas_h],
    ]])
    transformed = cv2.perspectiveTransform(points, h)[0]
    x1, y1 = np.floor(transformed.min(axis=0)).astype(int)
    x2, y2 = np.ceil(transformed.max(axis=0)).astype(int)
    return [
        int(np.clip(x1, 0, destination_w - 1)), int(np.clip(y1, 0, destination_h - 1)),
        int(np.clip(x2, 0, destination_w)), int(np.clip(y2, 0, destination_h)),
    ]


def rect_overlap(a: list[int], b: dict[str, Any]) -> dict[str, float | bool]:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = (int(b[key]) for key in ("left", "top", "right", "bottom"))
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    intersection = iw * ih
    group_area = max(1, (ax2 - ax1) * (ay2 - ay1))
    center_inside = ax1 <= (bx1 + bx2) / 2 <= ax2 and ay1 <= (by1 + by2) / 2 <= ay2
    # Proxy only: candidate centered in the group or covers >=5% of the group.
    hit = bool(center_inside or intersection / group_area >= 0.05)
    return {"intersection_pixels": intersection, "group_coverage": round(intersection / group_area, 4), "candidate_center_in_group": center_inside, "proxy_hit": hit}


def evidence(candidate: dict[str, Any]) -> tuple[float, float, float]:
    scores = candidate.get("evidence_scores", {})
    return (
        float(scores.get("traditional_p90_max", 0.0) or 0.0),
        float(scores.get("dino_p90_max", 0.0) or 0.0),
        float(scores.get("cross_evidence_pixel_ratio_max", 0.0) or 0.0),
    )


def keep_candidate(candidate: dict[str, Any], dino_cap: float | None, cross_cap: float | None) -> bool:
    if dino_cap is None or cross_cap is None:
        return True
    traditional_p90, dino_p90, cross_ratio = evidence(candidate)
    # Experimental terminal-texture guard: suppress only candidates whose
    # traditional edge response is high while both semantic and cross evidence
    # remain weak.  No candidate-generation parameter is altered.
    return not (traditional_p90 >= 150.0 and dino_p90 <= dino_cap and cross_ratio <= cross_cap)


def overlay(path: Path, image: np.ndarray, groups: list[dict[str, Any]], candidates: list[dict[str, Any]], title: str) -> None:
    drawing = image.copy()
    for group in groups:
        x1, y1, x2, y2 = group["mapped_reference_bbox"]
        cv2.rectangle(drawing, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(drawing, group["id"], (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2, cv2.LINE_AA)
    for index, candidate in enumerate(candidates, 1):
        cv2.rectangle(drawing, (candidate["left"], candidate["top"]), (candidate["right"], candidate["bottom"]), (255, 0, 255), 2)
        cv2.putText(drawing, f"C{index}", (candidate["left"], min(700, candidate["bottom"] + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(drawing, title, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(drawing, title, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 1, cv2.LINE_AA)
    ok, data = cv2.imencode(".png", drawing)
    if not ok:
        raise RuntimeError(f"Cannot encode {path}")
    data.tofile(str(path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "manual_review" / "candidate_filter_sweep_20260827")
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    standard = json.loads(STANDARD.read_text(encoding="utf-8"))
    controls = json.loads(CONTROL_SUMMARY.read_text(encoding="utf-8"))["cases"]
    reference = read_image(DESKTOP_CASE / "right.png")
    rois = [[0.01, 0.02, 0.99, 0.98]]

    positive_cases: list[dict[str, Any]] = []
    for item in standard["images"]:
        if not item.get("eligible_for_local_group_coverage"):
            continue
        inspection_path = DESKTOP_CASE / item["image"]
        inspection = read_image(inspection_path)
        aligned, alignment = perspective.automatic_homography(reference, inspection)
        if aligned is None:
            raise RuntimeError(f"Alignment rejected for {inspection_path.name}: {alignment.get('reason')}")
        h = replay_global_h(reference, inspection)
        replay = cv2.warpPerspective(inspection, h, (reference.shape[1], reference.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        replay_mae = float(np.mean(np.abs(replay.astype(np.int16) - aligned.astype(np.int16))))
        _, _, candidates = dino.dino_fused_regions(reference, aligned, rois)
        groups = []
        for group in item["groups"]:
            mapped = map_canvas_box_to_reference(group["bbox"], (inspection.shape[1], inspection.shape[0]), h)
            groups.append({"id": group["id"], "manual_canvas_bbox": group["bbox"], "mapped_reference_bbox": mapped, "description": group["description"]})
        record = {
            "image": item["image"], "alignment": alignment,
            "manual_transform_replay_mae_pixels": round(replay_mae, 5),
            "groups": groups, "candidates": candidates,
        }
        positive_cases.append(record)
        write_json(output / f"{inspection_path.stem}_baseline.json", record)
        overlay(output / f"{inspection_path.stem}_baseline.png", aligned, groups, candidates, f"{inspection_path.name} baseline: green=manual, magenta=candidate")

    cases: list[dict[str, Any]] = []
    for record in positive_cases:
        cases.append({"kind": "manual_local_change", "name": record["image"], "groups": record["groups"], "candidates": record["candidates"]})
    for control in controls:
        cases.append({"kind": "no_fault_planar_control", "name": control["name"], "groups": [], "candidates": control.get("review_regions", [])})

    variants = [("baseline", None, None)] + [(f"guard_d{d:.2f}_c{c:.2f}", d, c) for d in (0.10, 0.15, 0.20) for c in (0.10, 0.12, 0.15)]
    summary: list[dict[str, Any]] = []
    for name, dino_cap, cross_cap in variants:
        manual_hits = 0
        manual_total = 0
        control_counts: dict[str, int] = {}
        per_case = []
        for case in cases:
            kept = [candidate for candidate in case["candidates"] if keep_candidate(candidate, dino_cap, cross_cap)]
            if case["kind"] == "no_fault_planar_control":
                control_counts[case["name"]] = len(kept)
                continue
            group_results = []
            for group in case["groups"]:
                overlaps = [rect_overlap(group["mapped_reference_bbox"], candidate) for candidate in kept]
                hit = any(bool(value["proxy_hit"]) for value in overlaps)
                manual_hits += int(hit)
                manual_total += 1
                group_results.append({"group_id": group["id"], "proxy_hit": hit, "candidate_overlaps": overlaps})
            per_case.append({"image": case["name"], "kept_candidate_count": len(kept), "groups": group_results})
        summary.append({
            "variant": name,
            "experimental_guard": {"traditional_p90_min": 150.0, "dino_p90_max": dino_cap, "cross_evidence_ratio_max": cross_cap},
            "manual_groups_proxy_hit": f"{manual_hits}/{manual_total}",
            "manual_group_proxy_note": "center-in-group OR >=5% mapped-group overlap; mechanical proxy only, overlay still needs review.",
            "no_fault_control_candidate_counts": control_counts,
            "no_fault_control_total_candidates": sum(control_counts.values()),
            "manual_cases": per_case,
        })
    result = {
        "purpose": "Independent post-filter sweep; no mainline source or candidate generator changed.",
        "manual_standard": str(STANDARD),
        "controls": str(CONTROL_SUMMARY),
        "known_limits": [
            "Four manual groups are a compact fair regression set, not a field accuracy test.",
            "Planar controls have no inserted defects; their candidate counts are not a real-world false-positive rate.",
            "The H replay exists only to transform manual canvas coordinates and is checked against the accepted mainline-aligned pixels.",
        ],
        "positive_baselines": positive_cases,
        "variants": summary,
    }
    write_json(output / "summary.json", result)
    print(json.dumps([{key: row[key] for key in ("variant", "manual_groups_proxy_hit", "no_fault_control_total_candidates")} for row in summary], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
