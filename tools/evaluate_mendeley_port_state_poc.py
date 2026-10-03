"""Evaluate visible unplugged-port candidates on the held-out Mendeley test set."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


NAMES = {0: "unplugged_plug", 1: "unplugged_jack"}


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


def read_boxes(path: Path, width: int, height: int) -> list[dict[str, Any]]:
    boxes: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if not values:
            continue
        class_id = int(values[0])
        points = np.array([float(value) for value in values[1:]], dtype=np.float32).reshape(-1, 2)
        left, top = points.min(axis=0)
        right, bottom = points.max(axis=0)
        boxes.append({"class_id": class_id, "box_xyxy": [int(left * width), int(top * height), int(right * width), int(bottom * height)]})
    return boxes


def iou(left: list[int], right: list[int]) -> float:
    inter_left = max(left[0], right[0])
    inter_top = max(left[1], right[1])
    inter_right = min(left[2], right[2])
    inter_bottom = min(left[3], right[3])
    intersection = max(0, inter_right - inter_left) * max(0, inter_bottom - inter_top)
    union = (left[2] - left[0]) * (left[3] - left[1]) + (right[2] - right[0]) * (right[3] - right[1]) - intersection
    return intersection / union if union else 0.0


def greedy_match(truth: list[dict[str, Any]], predictions: list[dict[str, Any]], threshold: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    possible = []
    for truth_index, target in enumerate(truth):
        for prediction_index, prediction in enumerate(predictions):
            if target["class_id"] == prediction["class_id"]:
                possible.append((iou(target["box_xyxy"], prediction["box_xyxy"]), truth_index, prediction_index))
    matched_truth: set[int] = set()
    matched_predictions: set[int] = set()
    matches: list[tuple[int, int]] = []
    for overlap, truth_index, prediction_index in sorted(possible, reverse=True):
        if overlap < threshold or truth_index in matched_truth or prediction_index in matched_predictions:
            continue
        matched_truth.add(truth_index)
        matched_predictions.add(prediction_index)
        matches.append((truth_index, prediction_index))
    return matches, sorted(set(range(len(truth))) - matched_truth), sorted(set(range(len(predictions))) - matched_predictions)


def predictions_for(model: YOLO, image: np.ndarray, confidence: float, image_size: int) -> list[dict[str, Any]]:
    result = model.predict(image, imgsz=image_size, conf=confidence, device="cpu", verbose=False)[0]
    records: list[dict[str, Any]] = []
    for box in result.boxes:
        class_id = int(box.cls.item())
        if class_id not in NAMES:
            continue
        records.append(
            {
                "class_id": class_id,
                "confidence": round(float(box.conf.item()), 5),
                "box_xyxy": [int(value) for value in box.xyxy[0].tolist()],
            }
        )
    return records


def draw_evidence(image: np.ndarray, truth: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> np.ndarray:
    canvas = image.copy()
    for item in truth:
        left, top, right, bottom = item["box_xyxy"]
        cv2.rectangle(canvas, (left, top), (right, bottom), (255, 255, 0), 3)
        cv2.putText(canvas, f"GT {NAMES[item['class_id']]}", (left, max(24, top - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 0), 2, cv2.LINE_AA)
    for item in predictions:
        left, top, right, bottom = item["box_xyxy"]
        cv2.rectangle(canvas, (left, top), (right, bottom), (0, 0, 255), 3)
        cv2.putText(canvas, f"P {NAMES[item['class_id']]} {item['confidence']:.2f}", (left, min(canvas.shape[0] - 10, bottom + 28)), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 255), 2, cv2.LINE_AA)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.1, help="Low threshold is explicit because source boxes can be tiny.")
    parser.add_argument("--imgsz", type=int, default=960)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    images = sorted((args.dataset_dir / "images" / "test").glob("*.JPG"))
    labels = args.dataset_dir / "labels" / "test"
    if not images or not labels.is_dir():
        raise FileNotFoundError("expected images/test and labels/test in derived dataset")
    args.output.mkdir(parents=True)
    evidence_dir = args.output / "evidence"
    evidence_dir.mkdir()
    model = YOLO(str(args.weights))
    class_stats: dict[int, Counter[str]] = {class_id: Counter() for class_id in NAMES}
    summary = Counter()
    cases: list[dict[str, Any]] = []
    for image_path in images:
        image = read_image(image_path)
        truth = read_boxes(labels / f"{image_path.stem}.txt", image.shape[1], image.shape[0])
        predictions = predictions_for(model, image, args.confidence, args.imgsz)
        matches, missed, extra = greedy_match(truth, predictions, args.iou)
        for truth_index, _prediction_index in matches:
            class_stats[truth[truth_index]["class_id"]]["tp"] += 1
        for truth_index in missed:
            class_stats[truth[truth_index]["class_id"]]["fn"] += 1
        for prediction_index in extra:
            class_stats[predictions[prediction_index]["class_id"]]["fp"] += 1
        summary["images"] += 1
        summary["positive_images"] += int(bool(truth))
        summary["positive_images_hit"] += int(bool(truth) and bool(matches))
        summary["negative_images"] += int(not truth)
        summary["negative_images_with_candidate"] += int(not truth and bool(predictions))
        case = {
            "image": image_path.name,
            "truth_count": len(truth),
            "prediction_count": len(predictions),
            "matched_count": len(matches),
            "missed_count": len(missed),
            "extra_count": len(extra),
            "truth": truth,
            "predictions": predictions,
        }
        cases.append(case)
        write_image(evidence_dir / f"{image_path.stem}.jpg", draw_evidence(image, truth, predictions))

    metrics = {}
    for class_id, counts in class_stats.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        metrics[NAMES[class_id]] = {"tp": tp, "fp": fp, "fn": fn, "precision": round(precision, 5), "recall": round(recall, 5)}
    report = {
        "scope": "held-out source test split for fixed-Dell visible port-state POC; not independent field validation",
        "weights": str(args.weights),
        "settings": {"confidence": args.confidence, "iou": args.iou, "imgsz": args.imgsz},
        "image_summary": dict(summary),
        "class_metrics": metrics,
        "cases": cases,
        "limitations": [
            "The supplied train/val/test split may contain near-duplicate camera conditions.",
            "The training supervision consists of source rectangles converted to polygons, not true masks.",
            "This measures visible disconnected-port candidates only, not correct seating, cable route, or electrical continuity.",
        ],
    }
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"image_summary": report["image_summary"], "class_metrics": metrics, "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
