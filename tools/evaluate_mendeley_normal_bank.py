"""Evaluate a DINO spatial-descriptor normal-reference bank on Mendeley.

Only train01 normal images populate the bank.  A threshold is selected on
val01 with a fault-sensitivity floor and then applied once to frozen test01.
The result is an image-level review gate; existing difference candidates still
provide localization and remain human-review evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROTOTYPE = ROOT / "prototype"
for path in (ROOT, PROTOTYPE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dino_feature_diff import extract_features  # noqa: E402
from inspection_agent.normal_reference import (  # noqa: E402
    classification_metrics,
    descriptor_from_patch_features,
    nearest_normal_score,
    select_balanced_threshold,
)


DEFAULT_DATASET = (
    ROOT
    / "data"
    / "external_datasets"
    / "mendeley_electrical_wiring_faults"
    / "Predictive Maintenance for Electrical Wiring Faults"
)
FAULT_KINDS = {"damaged", "disconnected", "misrouted"}


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read image: {path}")
    return image


def kind_of(path: Path) -> str:
    return path.stem.split("_", 1)[0]


def cache_path_for(image_path: Path, cache_dir: Path) -> Path:
    stat = image_path.stat()
    identity = f"{image_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|spatial2"
    return cache_dir / f"{hashlib.sha256(identity.encode('utf-8')).hexdigest()}.npy"


def descriptor_for(image_path: Path, cache_dir: Path) -> tuple[np.ndarray, str]:
    cache_path = cache_path_for(image_path, cache_dir)
    if cache_path.is_file():
        return np.load(cache_path).astype(np.float32), "hit"
    features, _metadata = extract_features(read_image(image_path), cache_reference=False)
    descriptor = descriptor_from_patch_features(features, grid_size=2)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, descriptor)
    return descriptor, "miss"


def split_images(dataset: Path, split: str) -> list[Path]:
    return sorted((dataset / "images" / split).glob("*.JPG"))


def evaluate_split(
    paths: list[Path],
    bank: np.ndarray,
    bank_paths: list[Path],
    *,
    k: int,
    cache_dir: Path,
    progress_prefix: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, image_path in enumerate(paths, start=1):
        descriptor, cache_state = descriptor_for(image_path, cache_dir)
        score, neighbors = nearest_normal_score(descriptor, bank, k=k)
        for neighbor in neighbors:
            neighbor["image"] = bank_paths[neighbor.pop("index")].name
        record = {
            "image": image_path.name,
            "kind": kind_of(image_path),
            "is_fault": kind_of(image_path) in FAULT_KINDS,
            "normal_bank_score": score,
            "nearest_normal_references": neighbors,
        }
        records.append(record)
        print(
            f"[{progress_prefix} {index}/{len(paths)}] {image_path.name} "
            f"score={score:.6f} cache={cache_state}",
            flush=True,
        )
    return records


def attach_decisions(records: list[dict[str, Any]], threshold: float) -> None:
    for record in records:
        record["decision"] = (
            "possible_fault_manual_review"
            if record["normal_bank_score"] >= threshold
            else "normal_like_reference_bank"
        )
        record["automatic_fault_verdict"] = False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Reuse a descriptor cache across k-value audits (default: OUTPUT/descriptor_cache)",
    )
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--minimum-sensitivity", type=float, default=0.90)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or (args.output / "descriptor_cache")

    bank_paths = [path for path in split_images(args.dataset, "train01") if kind_of(path) == "normal"]
    bank_descriptors: list[np.ndarray] = []
    for index, image_path in enumerate(bank_paths, start=1):
        descriptor, cache_state = descriptor_for(image_path, cache_dir)
        bank_descriptors.append(descriptor)
        print(f"[bank {index}/{len(bank_paths)}] {image_path.name} cache={cache_state}", flush=True)
    bank = np.stack(bank_descriptors)

    validation = evaluate_split(
        split_images(args.dataset, "val01"),
        bank,
        bank_paths,
        k=args.k,
        cache_dir=cache_dir,
        progress_prefix="val",
    )
    selection = select_balanced_threshold(
        [record["normal_bank_score"] for record in validation],
        [record["is_fault"] for record in validation],
        minimum_sensitivity=args.minimum_sensitivity,
    )
    threshold = float(selection["threshold"])
    attach_decisions(validation, threshold)

    np.savez_compressed(
        args.output / "normal_bank_model.npz",
        descriptors=bank.astype(np.float32),
        reference_names=np.asarray([path.name for path in bank_paths]),
        threshold=np.asarray(threshold, dtype=np.float64),
        k=np.asarray(args.k, dtype=np.int32),
        schema_version=np.asarray(1, dtype=np.int32),
    )

    test = evaluate_split(
        split_images(args.dataset, "test01"),
        bank,
        bank_paths,
        k=args.k,
        cache_dir=cache_dir,
        progress_prefix="test",
    )
    attach_decisions(test, threshold)
    test_metrics = classification_metrics(
        [record["normal_bank_score"] for record in test],
        [record["is_fault"] for record in test],
        threshold,
    )
    report = {
        "schema_version": 1,
        "status": "ok",
        "mode": "dino_spatial_descriptor_multi_normal_reference_gate",
        "dataset": str(args.dataset),
        "protocol": {
            "normal_bank": "all train01 normal images",
            "normal_bank_size": len(bank_paths),
            "descriptor": "mean DINO patch token plus 2x2 spatial means, L2 normalized",
            "score": f"mean cosine distance to {args.k} nearest train01 normal descriptors",
            "threshold_selection": "val01 only",
            "test01_policy": "applied once after validation threshold was fixed",
            "minimum_validation_sensitivity": args.minimum_sensitivity,
        },
        "threshold_selection": selection,
        "model_artifact": "normal_bank_model.npz",
        "test_metrics": test_metrics,
        "validation": validation,
        "test": test,
        "evidence_boundary": [
            "This is an image-level normal-reference review gate, not a fault-type classifier.",
            "Existing visual candidates are still required to localize regions for human review.",
            "The dataset contains one controlled chassis and likely correlated captures; this is not field accuracy.",
            "test01 had been inspected in earlier experiments, so this is a frozen retrospective evaluation, not a pristine unseen holdout.",
        ],
    }
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "threshold": threshold,
        "validation_metrics": selection["metrics"],
        "test_metrics": test_metrics,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
