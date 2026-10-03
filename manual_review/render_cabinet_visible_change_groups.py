"""Render the cabinet visible-change-group draft for human review.

Green boxes are manually drafted connected visible-change groups. Yellow boxes
are existing main-pipeline candidates. The output is a review aid only: it is
not a Labelme export, training data, electrical truth, or a field metric.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch  # Must load before the perspective module imports PyQt on Windows.


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_DRAFT = Path(__file__).with_name("cabinet_visible_change_groups_draft.json")
DEFAULT_CANDIDATES = Path(os.environ["LOCALAPPDATA"]) / "Temp" / "cabinet_wrong_box_audit_20260826_v1" / "summary.json"
DEFAULT_OUTPUT = Path(os.environ["LOCALAPPDATA"]) / "Temp" / "cabinet_manual_group_draft_20260826_v1"

sys.path.insert(0, str(PROJECT / "prototype"))
import assembly_auto_review_robust_v3 as perspective  # noqa: E402


GREEN = (70, 220, 70)
YELLOW = (0, 220, 255)
WHITE = (245, 245, 245)
PANEL = (35, 35, 35)


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def intersect(first: list[int], second: list[int]) -> bool:
    return max(first[0], second[0]) < min(first[2], second[2]) and max(first[1], second[1]) < min(first[3], second[3])


def draw_box(image: np.ndarray, box: list[int], label: str, color: tuple[int, int, int]) -> None:
    left, top, right, bottom = (int(value) for value in box)
    cv2.rectangle(image, (left, top), (right, bottom), color, 3, cv2.LINE_AA)
    cv2.rectangle(image, (left, max(0, top - 22)), (left + 58, top), color, -1, cv2.LINE_AA)
    cv2.putText(image, label, (left + 4, top - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)


def card(reference: np.ndarray, aligned: np.ndarray, case: dict, candidates: list[list[int]]) -> np.ndarray:
    left, right = reference.copy(), aligned.copy()
    for group in case["groups"]:
        draw_box(left, group["bbox"], group["id"], GREEN)
        draw_box(right, group["bbox"], group["id"], GREEN)
    for index, box in enumerate(candidates, 1):
        draw_box(right, box, f"C{index}", YELLOW)
    joined = np.hstack([left, right])
    header = np.full((42, joined.shape[1], 3), PANEL, dtype=np.uint8)
    status = "EXCLUDED (ambiguous)" if case.get("exclude_from_group_metrics") else f"{len(case['groups'])} manual group(s)"
    title = f"{case['image']} | green=manual groups | yellow=main-pipeline candidates | {status}"
    cv2.putText(header, title, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, WHITE, 2, cv2.LINE_AA)
    return np.vstack([header, joined])


def make_page(cards: list[np.ndarray]) -> np.ndarray:
    scale = 0.39
    small = [cv2.resize(item, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) for item in cards]
    blank = np.zeros_like(small[0])
    while len(small) < 4:
        small.append(blank.copy())
    return np.vstack([np.hstack(small[:2]), np.hstack(small[2:4])])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft", type=Path, default=DEFAULT_DRAFT)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    draft = json.loads(args.draft.read_text(encoding="utf-8"))
    candidate_rows = json.loads(args.candidates.read_text(encoding="utf-8"))
    candidates_by_image = {
        row["id"]: [[item["left"], item["top"], item["right"], item["bottom"]] for item in row["candidates"]]
        for row in candidate_rows
    }
    reference = read_image(Path(draft["reference_image"]))
    image_root = Path(draft["reference_image"]).parent
    args.output.mkdir(parents=True, exist_ok=True)

    coverage_rows = []
    cards = []
    for case in draft["images"]:
        inspection = read_image(image_root / case["image"])
        aligned, alignment = perspective.automatic_homography(reference, inspection)
        if aligned is None:
            raise RuntimeError(f"Alignment failed for {case['image']}: {alignment}")
        candidate_boxes = candidates_by_image.get(case["image"], [])
        cards.append(card(reference, aligned, case, candidate_boxes))
        for group in case["groups"]:
            hits = [index for index, candidate in enumerate(candidate_boxes, 1) if intersect(group["bbox"], candidate)]
            coverage_rows.append({
                "image": case["image"],
                "group": group["id"],
                "confidence": group["confidence"],
                "candidate_indices_with_positive_overlap": hits,
                "covered": bool(hits),
                "exclude_from_metrics": bool(case.get("exclude_from_group_metrics")),
            })

    for page_index in range(0, len(cards), 4):
        page = make_page(cards[page_index:page_index + 4])
        output = args.output / f"page_{page_index // 4 + 1}.jpg"
        if not cv2.imwrite(str(output), page):
            raise RuntimeError(f"Cannot write {output}")

    covered = sum(row["covered"] for row in coverage_rows if not row["exclude_from_metrics"])
    eligible = sum(not row["exclude_from_metrics"] for row in coverage_rows)
    report = {
        "kind": "draft_manual_group_candidate_overlap_aid",
        "warning": "Manual boxes are a review draft and candidate boxes were cross-checked; do not use this as an unbiased formal metric until user confirmation.",
        "eligible_group_count": eligible,
        "groups_with_positive_candidate_overlap": covered,
        "rows": coverage_rows,
    }
    (args.output / "coverage_draft.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "eligible_groups": eligible, "covered_groups": covered}, ensure_ascii=False))


if __name__ == "__main__":
    main()
