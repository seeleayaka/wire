"""Audit whether nearby numbered normal captures inflate bank accuracy.

The reference bank still contains train01 normals only. At scoring time, a
normal whose numeric image id is within ``exclusion_radius`` of the query is
unavailable. Select the threshold using val01, then report test01 once.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inspection_agent.normal_reference import classification_metrics, select_balanced_threshold  # noqa: E402


def number(name: str) -> int:
    return int(Path(name).stem.rsplit("_", 1)[1])


def score_split(records: list[dict], bank: np.ndarray, bank_names: list[str], radius: int, k: int) -> list[dict]:
    out = []
    for record in records:
        image_number = number(record["image"])
        allowed = np.asarray([abs(number(name) - image_number) > radius for name in bank_names])
        query = np.asarray(record["descriptor"], dtype=np.float32)
        distances = np.clip(1.0 - bank[allowed] @ query, 0.0, 2.0)
        nearest = np.sort(distances)[:k]
        out.append({"image": record["image"], "kind": record["kind"], "is_fault": record["is_fault"], "score": float(nearest.mean()), "available_references": int(allowed.sum())})
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-report", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--radius", type=int, default=10)
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()
    if args.radius < 0:
        raise ValueError("radius must be nonnegative")
    from tools.evaluate_mendeley_normal_bank import cache_path_for

    source = json.loads(args.base_report.read_text(encoding="utf-8"))
    model = np.load(args.model)
    bank = model["descriptors"].astype(np.float32)
    names = [str(value) for value in model["reference_names"]]
    bank = bank / np.linalg.norm(bank, axis=1, keepdims=True)

    def load_records(split: str) -> list[dict]:
        rows = []
        for record in source["validation" if split == "val01" else "test"]:
            image_path = args.dataset / "images" / split / record["image"]
            descriptor = np.load(cache_path_for(image_path, args.cache_dir)).astype(np.float32)
            descriptor /= np.linalg.norm(descriptor)
            rows.append({**record, "descriptor": descriptor})
        return rows

    validation = score_split(load_records("val01"), bank, names, args.radius, args.k)
    selection = select_balanced_threshold([x["score"] for x in validation], [x["is_fault"] for x in validation], minimum_sensitivity=0.9)
    test = score_split(load_records("test01"), bank, names, args.radius, args.k)
    result = {
        "protocol": {"reference_split": "train01 normals", "threshold_split": "val01", "evaluation_split": "test01", "excluded_numeric_id_radius": args.radius, "k": args.k, "note": "Numeric proximity is a capture-correlation proxy, not a true session label."},
        "threshold_selection": selection,
        "test_metrics": classification_metrics([x["score"] for x in test], [x["is_fault"] for x in test], selection["threshold"]),
        "validation": validation,
        "test": test,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"threshold": selection["threshold"], "validation": selection["metrics"], "test": result["test_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
