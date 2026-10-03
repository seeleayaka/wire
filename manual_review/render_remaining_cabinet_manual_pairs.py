"""Render raw reference/inspection pairs and manual-only boxes from the remaining-cases standard."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

STANDARD = Path(__file__).with_name("cabinet_remaining_manual_visual_standard_v1.json")


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    document = json.loads(STANDARD.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    for material in document["materials"]:
        reference_path = Path(material["reference_image"])
        reference = read_image(reference_path)
        output_dir = args.output / f"material_{material['material']}"
        output_dir.mkdir(parents=True, exist_ok=True)
        for case in material["images"]:
            inspection_path = reference_path.parent / case["image"]
            inspection = cv2.resize(read_image(inspection_path), (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_AREA)
            rendered = inspection.copy()
            for group in case["groups"]:
                x1, y1, x2, y2 = group["bbox"]
                cv2.rectangle(rendered, (x1, y1), (x2, y2), (70, 220, 70), 2, cv2.LINE_AA)
                cv2.putText(rendered, group["id"], (x1 + 2, max(16, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (70, 220, 70), 1, cv2.LINE_AA)
            header = np.full((32, reference.shape[1] * 2, 3), 35, dtype=np.uint8)
            label = f"raw reference | raw inspection; status={case['status']}; green=manual visual group"
            cv2.putText(header, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (245, 245, 245), 1, cv2.LINE_AA)
            target = output_dir / f"{Path(case['image']).stem}_manual_raw_pair.png"
            ok, encoded = cv2.imencode(".png", np.vstack([header, np.hstack([reference, rendered])]))
            if not ok:
                raise RuntimeError(f"Cannot encode: {target}")
            encoded.tofile(str(target))
    print(args.output)


if __name__ == "__main__":
    main()
