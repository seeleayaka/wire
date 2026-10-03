"""Compare two localization reports and the union of their review evidence.

Each candidate set lives in its own reference-image coordinates. Targets are
paired by source-label order, and union IoU is the better overlap in either
coordinate frame. The union is a review policy, not a fused pixel mask.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(rows: list[dict]) -> dict:
    all_ious = [value for row in rows for value in row["best_target_ious"]]
    return {
        "fault_images": len(rows),
        "fault_images_with_target_overlap": sum(any(value > 0 for value in row["best_target_ious"]) for row in rows),
        "source_target_boxes": len(all_ious),
        "source_target_boxes_with_overlap": sum(value > 0 for value in all_ious),
        "source_target_boxes_iou_ge_0_1": sum(value >= 0.1 for value in all_ious),
        "mean_best_target_iou": sum(all_ious) / len(all_ious) if all_ious else None,
        "candidate_count": sum(row["candidate_count"] for row in rows),
        "candidates_per_fault_image": sum(row["candidate_count"] for row in rows) / len(rows) if rows else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    first = json.loads(args.first.read_text(encoding="utf-8"))
    second = json.loads(args.second.read_text(encoding="utf-8"))
    first_rows = {row["image"]: row for row in first["cases"] if row["kind"] != "normal"}
    second_rows = {row["image"]: row for row in second["cases"] if row["kind"] != "normal"}
    if set(first_rows) != set(second_rows):
        raise ValueError("reports must contain the same fault images")
    records = []
    for image in sorted(first_rows):
        a, b = first_rows[image], second_rows[image]
        if a["source_target_box_count"] != b["source_target_box_count"]:
            raise ValueError(f"source box count differs for {image}")
        if len(a["best_target_ious"]) != len(b["best_target_ious"]):
            raise ValueError(f"source box order differs for {image}")
        records.append({
            "image": image,
            "kind": a["kind"],
            "references": [a["reference"], b["reference"]],
            "candidate_count": len(a["candidates"]) + len(b["candidates"]),
            "best_target_ious": [max(x, y) for x, y in zip(a["best_target_ious"], b["best_target_ious"])],
        })
    report = {
        "protocol": "Union of two independently aligned review candidate sets; source labels used only after inference",
        "first_report": str(args.first),
        "second_report": str(args.second),
        "summary": summarize(records),
        "cases": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
