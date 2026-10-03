"""Apply a frozen Mendeley normal-reference gate to one image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROTOTYPE = ROOT / "prototype"
for path in (ROOT, PROTOTYPE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dino_feature_diff import extract_features  # noqa: E402
from inspection_agent.normal_reference import (  # noqa: E402
    descriptor_from_patch_features,
    nearest_normal_score,
)


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read image: {path}")
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    saved = np.load(args.model)
    if int(saved["schema_version"]) != 1:
        raise ValueError("normal-reference model schema_version must be 1")
    bank = saved["descriptors"].astype(np.float32)
    reference_names = [str(name) for name in saved["reference_names"].tolist()]
    threshold = float(saved["threshold"])
    k = int(saved["k"])
    features, feature_metadata = extract_features(read_image(args.image), cache_reference=False)
    descriptor = descriptor_from_patch_features(features, grid_size=2)
    score, neighbors = nearest_normal_score(descriptor, bank, k=k)
    for neighbor in neighbors:
        neighbor["image"] = reference_names[neighbor.pop("index")]
    result = {
        "schema_version": 1,
        "image": str(args.image),
        "normal_bank_score": score,
        "threshold": threshold,
        "decision": (
            "possible_fault_manual_review" if score >= threshold else "normal_like_reference_bank"
        ),
        "automatic_fault_verdict": False,
        "nearest_normal_references": neighbors,
        "feature_metadata": feature_metadata,
        "evidence_boundary": "dataset-domain image-level review gate; not fault type, topology, or field accuracy",
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
