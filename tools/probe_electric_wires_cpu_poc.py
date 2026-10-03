"""Run a trained Electric Wires POC model on untouched PC/cabinet target images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sources", nargs="+", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.01)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    missing = [str(path) for path in args.sources if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing sources: " + ", ".join(missing))
    model = YOLO(str(args.weights))
    results = model.predict(
        source=[str(path) for path in args.sources],
        device="cpu",
        imgsz=args.imgsz,
        conf=args.conf,
        max_det=30,
        retina_masks=True,
        save=True,
        project=str(args.output_dir),
        name=f"targets_conf_{args.conf:g}",
        exist_ok=True,
        verbose=False,
    )
    report: list[dict] = []
    for result in results:
        confidences = [] if result.boxes is None else [round(float(value), 5) for value in result.boxes.conf.cpu().tolist()]
        areas = []
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy().astype(bool)
            areas = [round(float(mask.mean()), 6) for mask in masks]
        report.append(
            {
                "source": str(result.path),
                "detections": len(confidences),
                "confidences": confidences,
                "mask_area_fractions": areas,
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / f"targets_conf_{args.conf:g}_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
