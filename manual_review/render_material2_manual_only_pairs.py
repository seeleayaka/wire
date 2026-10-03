"""Render selected material-2 raw pairs with manual boxes only; no registration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"C:\Users\HUAWEI\Desktop\案例\线材柜子\2")
STANDARD = Path(__file__).with_name("cabinet_material2_qualitative_manual_standard_v1.json")


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot read {path}")
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    document = json.loads(STANDARD.read_text(encoding="utf-8"))
    reference = read_image(ROOT / "2.png")
    args.output.mkdir(parents=True, exist_ok=True)
    for case in document["images"]:
        inspection = cv2.resize(read_image(ROOT / case["image"]), (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_AREA)
        for group in case["groups"]:
            x1, y1, x2, y2 = group["bbox"]
            cv2.rectangle(inspection, (x1, y1), (x2, y2), (70, 220, 70), 3, cv2.LINE_AA)
            cv2.putText(inspection, group["id"], (x1 + 3, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (70, 220, 70), 2, cv2.LINE_AA)
        header = np.full((34, reference.shape[1] * 2, 3), 35, dtype=np.uint8)
        cv2.putText(header, "raw reference | raw inspection; green = manual visual group only", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (245, 245, 245), 1, cv2.LINE_AA)
        target = args.output / (Path(case["image"]).stem + "_manual_raw_pair.png")
        ok, encoded = cv2.imencode(".png", np.vstack([header, np.hstack([reference, inspection])]))
        if not ok:
            raise RuntimeError(f"Cannot encode {target}")
        encoded.tofile(str(target))
    print(args.output)


if __name__ == "__main__":
    main()
