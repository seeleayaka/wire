"""Replay material-2 qualitative manual groups through the unmodified review chain."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Importing the helper imports torch before the PyQt-owning review modules.
from cabinet_candidate_filter_sweep import (
    dino, perspective, read_image, replay_global_h, map_canvas_box_to_reference,
    rect_overlap, keep_candidate, overlay, write_json,
)

ROOT = Path(r"C:\Users\HUAWEI\Desktop\案例\线材柜子\2")
STANDARD = Path(__file__).with_name("cabinet_material2_qualitative_manual_standard_v1.json")


def group_coverage(groups: list[dict], candidates: list[dict]) -> list[dict]:
    results = []
    for group in groups:
        overlaps = [rect_overlap(group["mapped_reference_bbox"], candidate) for candidate in candidates]
        results.append({
            "id": group["id"],
            "candidate_proxy_coverage": any(bool(item["proxy_hit"]) for item in overlaps),
            "candidate_overlaps": overlaps,
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("cabinet_material2_qualitative_replay_20260827"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    standard = json.loads(STANDARD.read_text(encoding="utf-8"))
    reference = read_image(ROOT / "2.png")
    canvas_size = tuple(standard["coordinate_space"]["canvas_size"])
    rois = [[0.01, 0.02, 0.99, 0.98]]
    cases = []
    for case in standard["images"]:
        inspection = read_image(ROOT / case["image"])
        aligned, alignment = perspective.automatic_homography(reference, inspection)
        if aligned is None:
            cases.append({"image": case["image"], "alignment": alignment, "decision": "alignment_rejected"})
            continue
        h = replay_global_h(reference, inspection)
        replay = __import__("cv2").warpPerspective(inspection, h, (reference.shape[1], reference.shape[0]), flags=__import__("cv2").INTER_LINEAR, borderMode=__import__("cv2").BORDER_CONSTANT)
        replay_mae = float(__import__("numpy").mean(__import__("numpy").abs(replay.astype(__import__("numpy").int16) - aligned.astype(__import__("numpy").int16))))
        _, _, candidates = dino.dino_fused_regions(reference, aligned, rois)
        groups = []
        for group in case["groups"]:
            groups.append({
                "id": group["id"], "manual_canvas_bbox": group["bbox"], "description": group["description"],
                "mapped_reference_bbox": map_canvas_box_to_reference(
                    group["bbox"], (inspection.shape[1], inspection.shape[0]), h,
                    canvas_size=canvas_size, destination_size=(reference.shape[1], reference.shape[0]),
                ),
            })
        guarded = [candidate for candidate in candidates if keep_candidate(candidate, 0.20, 0.20)]
        record = {
            "image": case["image"], "alignment": alignment,
            "manual_transform_replay_mae_pixels": round(replay_mae, 5), "groups": groups,
            "baseline_candidates": candidates, "guarded_candidates": guarded,
            "baseline_group_coverage": group_coverage(groups, candidates),
            "guarded_group_coverage": group_coverage(groups, guarded),
        }
        cases.append(record)
        stem = Path(case["image"]).stem
        write_json(args.output / f"{stem}_report.json", record)
        overlay(args.output / f"{stem}_baseline.png", aligned, groups, candidates, f"{case['image']} baseline; green=manual, magenta=candidate")
        overlay(args.output / f"{stem}_guarded.png", aligned, groups, guarded, f"{case['image']} guarded; green=manual, magenta=candidate")
    result = {
        "purpose": "Qualitative recall replay only. The manual groups were drawn before candidate replay; no score is a production metric.",
        "experimental_guard": "suppress only traditional_p90>=150 AND dino_p90<=0.20 AND cross_ratio<=0.20",
        "cases": cases,
    }
    write_json(args.output / "summary.json", result)
    for case in cases:
        if case.get("decision") == "alignment_rejected":
            print(case["image"], "alignment_rejected")
            continue
        baseline = sum(item["candidate_proxy_coverage"] for item in case["baseline_group_coverage"])
        guarded = sum(item["candidate_proxy_coverage"] for item in case["guarded_group_coverage"])
        print(case["image"], f"baseline={baseline}/{len(case['groups'])} ({len(case['baseline_candidates'])} candidates)", f"guarded={guarded}/{len(case['groups'])} ({len(case['guarded_candidates'])} candidates)")


if __name__ == "__main__":
    main()
