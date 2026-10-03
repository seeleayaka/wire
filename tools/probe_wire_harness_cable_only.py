"""Probe an external wire-harness segmentation model on cabinet images.

This tool is diagnostic only.  It limits inference to the two Cable classes
and writes masks plus a compact JSON summary outside the production flow.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


SOURCE_DIR = Path(r"C:\Users\HUAWEI\Desktop\案例\线材柜子\1")
MODEL_PATH = Path(r"E:\wire_harness_training_bundle\weights\wire_harness_v1_b32_best.pt")
OUTPUT_DIR = Path(r"E:\PythonProject10\output\wire_harness_cable_only_probe_20260824")
CABLE_CLASS_IDS = (0, 7)
LOW_CONFIDENCE_FLOOR = 0.05
REPORT_CONFIDENCE = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--overlay-limit",
        type=int,
        default=None,
        help="Maximum number of report-threshold mask overlays to save; omit for all.",
    )
    parser.add_argument(
        "--image",
        action="append",
        type=Path,
        default=None,
        help="Explicit image path to probe; repeat this option for a balanced small sample.",
    )
    return parser.parse_args()


def main(
    source_dir: Path,
    output_dir: Path,
    overlay_limit: int | None,
    selected_images: list[Path] | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = output_dir / "conf_025_masks"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    images = (
        sorted(selected_images)
        if selected_images
        else sorted(
            path
            for path in source_dir.rglob("*")
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        )
    )
    model = YOLO(str(MODEL_PATH))
    rows: list[dict[str, object]] = []
    overlays_saved = 0

    for image_path in images:
        result = model.predict(
            source=str(image_path),
            imgsz=1024,
            conf=LOW_CONFIDENCE_FLOOR,
            iou=0.70,
            classes=list(CABLE_CLASS_IDS),
            device="cpu",
            verbose=False,
            save=False,
        )[0]

        detections: list[dict[str, object]] = []
        if result.boxes is not None:
            for index, box in enumerate(result.boxes):
                class_id = int(box.cls.item())
                polygon: list[list[float]] | None = None
                if result.masks is not None and index < len(result.masks.xy):
                    polygon = result.masks.xy[index].round(1).tolist()
                detections.append(
                    {
                        "class_id": class_id,
                        "class_name": result.names[class_id],
                        "confidence": round(float(box.conf.item()), 4),
                        "xyxy": [round(float(value), 1) for value in box.xyxy[0].tolist()],
                        "polygon": polygon,
                    }
                )

        confident = [item for item in detections if item["confidence"] >= REPORT_CONFIDENCE]
        if confident and (overlay_limit is None or overlays_saved < overlay_limit):
            canvas = result.orig_img.copy()
            overlay = canvas.copy()
            for item in confident:
                polygon = item["polygon"]
                if not polygon:
                    continue
                points = np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)
                cv2.fillPoly(overlay, [points], color=(0, 180, 0))
                cv2.polylines(canvas, [points], isClosed=True, color=(0, 255, 0), thickness=2)
                x1, y1, _, _ = (int(value) for value in item["xyxy"])
                cv2.putText(
                    canvas,
                    f"{item['class_name']} {item['confidence']:.2f}",
                    (x1, max(22, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            canvas = cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0)
            relative = image_path.relative_to(source_dir).with_suffix(".jpg")
            overlay_path = overlay_dir / relative
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(overlay_path), canvas)
            overlays_saved += 1

        rows.append(
            {
                "image": str(image_path.relative_to(source_dir)),
                "image_size": [int(result.orig_shape[1]), int(result.orig_shape[0])],
                "detections_at_005": len(detections),
                "detections_at_025": len(confident),
                "max_confidence": max((item["confidence"] for item in detections), default=0.0),
                "detections": detections,
            }
        )

    summary = {
        "purpose": "Cable-only diagnostic probe; no production pipeline was changed.",
        "model": str(MODEL_PATH),
        "source": str(source_dir),
        "classes_used": {"0": "AJ20_D6_Cable", "7": "MHEV_N11_Cable"},
        "settings": {
            "imgsz": 1024,
            "confidence_floor": LOW_CONFIDENCE_FLOOR,
            "report_confidence": REPORT_CONFIDENCE,
            "iou": 0.70,
            "device": "cpu",
        },
        "images": rows,
        "aggregate": {
            "images": len(rows),
            "images_with_cable_at_005": sum(row["detections_at_005"] > 0 for row in rows),
            "images_with_cable_at_025": sum(row["detections_at_025"] > 0 for row in rows),
            "detections_at_005": sum(row["detections_at_005"] for row in rows),
            "detections_at_025": sum(row["detections_at_025"] for row in rows),
            "max_confidence_overall": max((row["max_confidence"] for row in rows), default=0.0),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary["aggregate"], ensure_ascii=False))
    print(output_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args.source, args.output, args.overlay_limit, args.image)
