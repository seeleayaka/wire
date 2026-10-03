"""Local inference for the fixed-Dell visible unplugged-port POC.

This program detects only two visibly labelled conditions: a detached plug or an
empty/detached jack.  It never asserts electrical continuity, full cable routing,
or that a frame without candidates is correctly assembled.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


CLASS_NAMES = {0: "unplugged_plug", 1: "unplugged_jack"}
CLASS_COLORS = {0: (80, 130, 255), 1: (225, 80, 230)}


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(path.suffix or ".jpg", image)
    if not ok:
        raise ValueError(f"cannot encode {path}")
    encoded.tofile(str(path))


def inspect(image: np.ndarray, weights: Path, confidence: float, image_size: int) -> tuple[np.ndarray, list[dict[str, object]]]:
    result = YOLO(str(weights)).predict(image, imgsz=image_size, conf=confidence, device="cpu", verbose=False)[0]
    overlay = image.copy()
    records: list[dict[str, object]] = []
    for box in result.boxes:
        class_id = int(box.cls.item())
        if class_id not in CLASS_NAMES:
            continue
        left, top, right, bottom = (int(value) for value in box.xyxy[0].tolist())
        confidence_value = round(float(box.conf.item()), 5)
        label = CLASS_NAMES[class_id]
        color = CLASS_COLORS[class_id]
        cv2.rectangle(overlay, (left, top), (right, bottom), color, 3)
        cv2.putText(overlay, f"{label} {confidence_value:.2f}", (left, max(24, top - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)
        records.append({"class_id": class_id, "class": label, "confidence": confidence_value, "box_xyxy": [left, top, right, bottom]})
    return overlay, records


def build_report(source: Path, weights: Path, confidence: float, records: list[dict[str, object]]) -> dict[str, object]:
    return {
        "scope": "fixed Dell chassis, visible port-state POC",
        "source": str(source),
        "weights": str(weights),
        "confidence_threshold": confidence,
        "candidates": records,
        "decision": "possible_unseated_port_manual_review" if records else "no_unseated_port_candidate_not_verified",
        "limitations": [
            "A no-candidate result is not proof that every plug is correctly seated.",
            "No electrical continuity, cable end-to-end route, or hidden connector state is inferred.",
            "Weights require separate hold-out and manual visual validation before any operational use.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=960)
    args = parser.parse_args()
    if not args.image.is_file() or not args.weights.is_file():
        raise FileNotFoundError("image or trained weights missing")
    args.output.mkdir(parents=True, exist_ok=False)
    overlay, records = inspect(read_image(args.image), args.weights, args.confidence, args.imgsz)
    write_image(args.output / "port_candidates.jpg", overlay)
    report = build_report(args.image, args.weights, args.confidence, records)
    report["artifacts"] = {"overlay": str(args.output / "port_candidates.jpg")}
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": report["decision"], "candidate_count": len(records), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
