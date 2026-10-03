"""Held-out evaluation for the fixed-Dell local empty-jack classifier."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from train_mendeley_jack_roi_classifier import RoiCropDataset, TinyRoiClassifier, load_records


def binary_metrics(targets: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    tp = int(np.logical_and(predicted, targets == 1).sum())
    fp = int(np.logical_and(predicted, targets == 0).sum())
    fn = int(np.logical_and(~predicted, targets == 1).sum())
    tn = int(np.logical_and(~predicted, targets == 0).sum())
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(tp / (tp + fp), 5) if tp + fp else 0.0,
        "recall": round(tp / (tp + fn), 5) if tp + fn else 0.0,
        "specificity": round(tn / (tn + fp), 5) if tn + fp else 0.0,
    }


def read_source_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read {path}")
    return image


def write_source_image(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise ValueError(f"cannot encode {path}")
    encoded.tofile(str(path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    recipe = json.loads((args.dataset_dir / "recipe.json").read_text(encoding="utf-8"))
    positive_name = str(recipe["class_names"]["1"])
    score_key = f"score_{positive_name}"
    payload = torch.load(args.weights, map_location="cpu", weights_only=False)
    threshold = float(payload["recommended_threshold"] if args.threshold is None else args.threshold)
    model = TinyRoiClassifier(int(payload["roi_count"]))
    model.load_state_dict(payload["model_state"])
    model.eval()
    records = load_records(args.dataset_dir, "test")
    loader = DataLoader(RoiCropDataset(args.dataset_dir, records), batch_size=32, shuffle=False, num_workers=0)
    scores: list[float] = []
    with torch.no_grad():
        for images, _labels, roi_indices in loader:
            scores.extend(torch.softmax(model(images, roi_indices), dim=1)[:, 1].numpy().tolist())
    targets = np.asarray([record.label for record in records], dtype=np.int64)
    predicted = np.asarray(scores) >= threshold
    args.output.mkdir(parents=True)
    per_image: dict[str, list[dict[str, object]]] = defaultdict(list)
    per_roi: dict[str, list[int]] = defaultdict(list)
    for record, score, guess in zip(records, scores, predicted, strict=True):
        image = Path(record.crop).stem + ".JPG"
        roi = recipe["candidate_rois"][record.roi_index]
        item = {"roi_id": roi["candidate_id"], "label": record.label, score_key: round(float(score), 6), "candidate": bool(guess)}
        per_image[image].append(item)
        per_roi[roi["candidate_id"]].append(len(per_image[image]) - 1)
    image_truth = np.asarray([any(item["label"] for item in items) for items in per_image.values()])
    image_predicted = np.asarray([any(item["candidate"] for item in items) for items in per_image.values()])
    evidence_dir = args.output / "evidence"
    evidence_dir.mkdir()
    for image_name, items in per_image.items():
        image = read_source_image(args.source_dataset / "images" / "test01" / image_name)
        height, width = image.shape[:2]
        for roi, item in zip(recipe["candidate_rois"], items, strict=True):
            left, top, right, bottom = roi["roi_normalized_xyxy"]
            x1, y1, x2, y2 = int(left * width), int(top * height), int(right * width), int(bottom * height)
            color = (0, 200, 0) if item["label"] else (100, 100, 100)
            if item["candidate"]:
                color = (0, 0, 255)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
            cv2.putText(image, f"{item['roi_id']} p={item[score_key]:.2f}", (x1, max(28, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        write_source_image(evidence_dir / f"{Path(image_name).stem}.jpg", image)
    report = {
        "scope": f"held-out source test split for {recipe['purpose']}; not independent field validation",
        "weights": str(args.weights), "threshold": threshold,
        "crop_metrics": binary_metrics(targets, predicted),
        "image_metrics": binary_metrics(image_truth.astype(np.int64), image_predicted),
        "cases": dict(per_image),
        "limitations": [
            "ROI positions are valid only for this photographed Dell chassis and viewpoint.",
            f"A negative result means no source-style {positive_name} candidate, not proof of correct seating, end-to-end route, or electrical continuity.",
            "The supplied split shares a tightly controlled fixture and is not field validation.",
        ],
    }
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"crop_metrics": report["crop_metrics"], "image_metrics": report["image_metrics"], "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
