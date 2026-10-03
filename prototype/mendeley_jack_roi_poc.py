"""Run the fixed-Dell local empty-jack POC on one chassis image.

The only positive output is a *visible empty-jack candidate* for manual review.
It never asserts that an unflagged connector is fully seated, correctly routed,
or electrically continuous.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from prepare_mendeley_jack_roi_dataset import crop_roi, read_image, write_image  # noqa: E402
from train_mendeley_jack_roi_classifier import TinyRoiClassifier  # noqa: E402


def build_report(image: Path, weights: Path, threshold: float, candidates: list[dict[str, object]], positive_name: str = "visible_empty_jack") -> dict[str, object]:
    return {
        "scope": f"fixed-Dell {positive_name} candidate POC only",
        "image": str(image),
        "weights": str(weights),
        "threshold": threshold,
        "decision": f"possible_{positive_name}_manual_review" if candidates else f"no_{positive_name}_candidate_not_verified",
        "candidates": candidates,
        "limitations": [
            "Candidate ROI IDs are geometric fixed-location IDs, not physical connector names.",
            "No candidate is not proof that a connector is seated, that its cable route is correct, or that it is electrically continuous.",
            "This model is only valid for the same Dell chassis and a closely matched viewpoint.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    positive_name = str(recipe["class_names"]["1"])
    score_key = f"score_{positive_name}"
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
    threshold = float(checkpoint["recommended_threshold"] if args.threshold is None else args.threshold)
    model = TinyRoiClassifier(int(checkpoint["roi_count"]))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    image = read_image(args.image)
    candidates: list[dict[str, object]] = []
    scores: list[float] = []
    with torch.no_grad():
        for roi_index, roi_record in enumerate(recipe["candidate_rois"]):
            crop = crop_roi(image, roi_record["roi_normalized_xyxy"], float(recipe["context_scale"]), int(recipe["output_size"]))
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float().div_(255.0).unsqueeze(0)
            score = float(torch.softmax(model(tensor, torch.tensor([roi_index])), dim=1)[0, 1])
            scores.append(score)
            if score >= threshold:
                candidates.append({"roi_id": roi_record["candidate_id"], score_key: round(score, 6)})
    overlay = image.copy()
    height, width = overlay.shape[:2]
    candidate_ids = {str(item["roi_id"]) for item in candidates}
    for roi_record, score in zip(recipe["candidate_rois"], scores, strict=True):
        left, top, right, bottom = roi_record["roi_normalized_xyxy"]
        x1, y1, x2, y2 = int(left * width), int(top * height), int(right * width), int(bottom * height)
        selected = str(roi_record["candidate_id"]) in candidate_ids
        color = (0, 0, 255) if selected else (90, 90, 90)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3 if selected else 1)
        if selected:
            cv2.putText(overlay, f"{roi_record['candidate_id']} p={score:.2f}", (x1, max(28, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    args.output.mkdir(parents=True)
    write_image(args.output / "overlay.jpg", overlay)
    report = build_report(args.image, args.weights, threshold, candidates, positive_name)
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": report["decision"], "candidate_count": len(candidates), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
