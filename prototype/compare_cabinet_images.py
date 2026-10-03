"""Compare the newest cabinet images in one folder against the DINO recipe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluate_noise_regression import candidates_for, load_cabinet_recipe


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def newest_images(directory: Path, count: int) -> list[Path]:
    images = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(images, key=lambda path: path.stat().st_mtime, reverse=True)[:count]


def compact_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        key: candidate[key]
        for key in (
            "left",
            "top",
            "right",
            "bottom",
            "area",
            "difference_score",
            "confidence",
            "comparison_mode",
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not args.directory.is_dir():
        raise NotADirectoryError(args.directory)
    if args.count < 1:
        raise ValueError("--count must be positive")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite report: {args.output}")

    reference, rois, _smoke_images = load_cabinet_recipe()
    images = newest_images(args.directory, args.count)
    if len(images) < args.count:
        raise ValueError(f"expected {args.count} images in {args.directory}, found {len(images)}")

    cases: list[dict[str, Any]] = []
    for image in images:
        result = candidates_for(reference, image, rois, real_pipeline=True)
        diagnostics = result.get("local_alignment", [])
        dino_error = diagnostics[0].get("dino_error") if diagnostics else None
        alignment_quality = result["alignment"].get("alignment_quality", {})
        cases.append(
            {
                "image": str(image),
                "alignment_reliable": bool(alignment_quality.get("reliable", False)),
                "alignment_reason": alignment_quality.get("reason"),
                "dino_error": dino_error,
                "candidate_count": len(result["candidates"]),
                "candidates": [compact_candidate(item) for item in result["candidates"]],
            }
        )

    report = {
        "reference": str(reference),
        "check_rois": rois,
        "candidate_contract": "possible_difference_manual_review",
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "cases": [
                    {
                        "name": Path(case["image"]).name,
                        "aligned": case["alignment_reliable"],
                        "dino_error": case["dino_error"],
                        "candidate_count": case["candidate_count"],
                    }
                    for case in cases
                ],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
