"""Dataset-bounded semantic cable segmentation baseline for MovingCables.

The first half of one clip fits a balanced HSV histogram classifier, the
second half selects threshold/morphology, and a different clip is evaluated
once.  This is a reproducible dataset baseline, not a cabinet model.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path, PurePosixPath
import tarfile
from typing import Any, Iterable

import cv2
import numpy as np


H_BINS = 36
S_BINS = 16
V_BINS = 16
TOTAL_BINS = H_BINS * S_BINS * V_BINS


class MovingCablesArchive:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.package = tarfile.open(path, "r")
        self.members = {PurePosixPath(member.name): member for member in self.package.getmembers() if member.isfile()}

    def close(self) -> None:
        self.package.close()

    def __enter__(self) -> "MovingCablesArchive":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def frames(self, image_type: str, clip: str) -> list[str]:
        prefix = PurePosixPath("MovingCables/sampled_compositions_small/test") / image_type / clip
        return sorted(path.name for path in self.members if path.parent == prefix and path.suffix.lower() == ".png")

    def image(self, image_type: str, clip: str, filename: str) -> np.ndarray:
        member_path = PurePosixPath("MovingCables/sampled_compositions_small/test") / image_type / clip / filename
        member = self.members.get(member_path)
        if member is None:
            raise FileNotFoundError(str(member_path))
        stream = self.package.extractfile(member)
        if stream is None:
            raise OSError(f"cannot read {member_path}")
        data = np.frombuffer(stream.read(), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise OSError(f"cannot decode {member_path}")
        return image

    def pair(self, clip: str, filename: str) -> tuple[np.ndarray, np.ndarray]:
        rgb = self.image("rgb_clips", clip, filename)
        flow = self.image("normal_flow_first_back", clip, filename)
        if flow.dtype != np.uint16 or flow.ndim != 3 or flow.shape[2] != 3:
            raise ValueError(f"unexpected normal-flow format for {clip}/{filename}: {flow.shape} {flow.dtype}")
        labels = flow[..., 0]
        if int(labels.max()) >= 32768:
            raise ValueError("OpenCV channel 0 is not the expected instance-label channel")
        return rgb, labels > 0


def _bin_indices(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h = np.minimum((hsv[..., 0].astype(np.int32) * H_BINS) // 180, H_BINS - 1)
    s = np.minimum((hsv[..., 1].astype(np.int32) * S_BINS) // 256, S_BINS - 1)
    v = np.minimum((hsv[..., 2].astype(np.int32) * V_BINS) // 256, V_BINS - 1)
    return (h * S_BINS + s) * V_BINS + v


def _fit_histogram(
    archive: MovingCablesArchive,
    clip: str,
    frames: Iterable[str],
    *,
    stride: int = 2,
    alpha: float = 1.0,
) -> np.ndarray:
    positive = np.full(TOTAL_BINS, alpha, dtype=np.float64)
    negative = np.full(TOTAL_BINS, alpha, dtype=np.float64)
    for filename in frames:
        rgb, ground_truth = archive.pair(clip, filename)
        bins = _bin_indices(rgb)[::stride, ::stride].reshape(-1)
        target = ground_truth[::stride, ::stride].reshape(-1)
        positive += np.bincount(bins[target], minlength=TOTAL_BINS)
        negative += np.bincount(bins[~target], minlength=TOTAL_BINS)
    positive /= positive.sum()
    negative /= negative.sum()
    return np.log(positive) - np.log(negative)


def _postprocess(mask: np.ndarray, open_size: int, close_size: int) -> np.ndarray:
    result = mask.astype(np.uint8) * 255
    if close_size:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
    if open_size:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
    return result > 0


def _counts(prediction: np.ndarray, target: np.ndarray) -> tuple[int, int, int]:
    return (
        int(np.count_nonzero(prediction & target)),
        int(np.count_nonzero(prediction & ~target)),
        int(np.count_nonzero(~prediction & target)),
    )


def _metrics(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "iou": iou, "f1": f1}


def _predict(rgb: np.ndarray, lookup: np.ndarray, threshold: float, open_size: int, close_size: int) -> np.ndarray:
    score = lookup[_bin_indices(rgb)]
    return _postprocess(score >= threshold, open_size, close_size)


def _evaluate(
    archive: MovingCablesArchive,
    clip: str,
    frames: Iterable[str],
    lookup: np.ndarray,
    *,
    threshold: float,
    open_size: int,
    close_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    totals = [0, 0, 0]
    per_frame: list[dict[str, Any]] = []
    for filename in frames:
        rgb, target = archive.pair(clip, filename)
        prediction = _predict(rgb, lookup, threshold, open_size, close_size)
        tp, fp, fn = _counts(prediction, target)
        totals[0] += tp
        totals[1] += fp
        totals[2] += fn
        per_frame.append({"frame": filename, **_metrics(tp, fp, fn)})
    aggregate = _metrics(*totals)
    frame_ious = [item["iou"] for item in per_frame]
    aggregate.update(
        {
            "frame_count": len(per_frame),
            "mean_frame_iou": float(np.mean(frame_ious)),
            "median_frame_iou": float(np.median(frame_ious)),
            "min_frame_iou": float(np.min(frame_ious)),
            "max_frame_iou": float(np.max(frame_ious)),
        }
    )
    return aggregate, per_frame


def _overlay(rgb: np.ndarray, target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    panels: list[np.ndarray] = []
    for title, mask, color in (
        ("RGB", None, None),
        ("GROUND TRUTH", target, (0, 220, 0)),
        ("PREDICTION", prediction, (0, 0, 240)),
    ):
        panel = rgb.copy()
        if mask is not None and color is not None:
            tint = np.zeros_like(panel)
            tint[mask] = color
            panel = cv2.addWeighted(panel, 0.72, tint, 0.28, 0)
        cv2.rectangle(panel, (0, 0), (220, 32), (0, 0, 0), -1)
        cv2.putText(panel, title, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(panel)
    return np.concatenate(panels, axis=1)


def evaluate(archive_path: Path, calibration_clip: str, evaluation_clip: str, output_dir: Path) -> dict[str, Any]:
    with MovingCablesArchive(archive_path) as archive:
        calibration_frames = archive.frames("rgb_clips", calibration_clip)
        evaluation_frames = archive.frames("rgb_clips", evaluation_clip)
        if len(calibration_frames) < 4 or not evaluation_frames:
            raise ValueError("both clips must contain enough frames")
        paired_calibration = set(calibration_frames).intersection(archive.frames("normal_flow_first_back", calibration_clip))
        paired_evaluation = set(evaluation_frames).intersection(archive.frames("normal_flow_first_back", evaluation_clip))
        calibration_frames = sorted(paired_calibration)
        evaluation_frames = sorted(paired_evaluation)
        split = len(calibration_frames) // 2
        fit_frames = calibration_frames[:split]
        validation_frames = calibration_frames[split:]
        lookup = _fit_histogram(archive, calibration_clip, fit_frames)

        candidates: list[dict[str, Any]] = []
        thresholds = np.linspace(-3.0, 3.0, 13)
        morphology = ((0, 0), (0, 5), (3, 5))
        for threshold in thresholds:
            for open_size, close_size in morphology:
                metrics, _ = _evaluate(
                    archive,
                    calibration_clip,
                    validation_frames,
                    lookup,
                    threshold=float(threshold),
                    open_size=open_size,
                    close_size=close_size,
                )
                candidates.append(
                    {
                        "threshold": float(threshold),
                        "open_size": open_size,
                        "close_size": close_size,
                        **metrics,
                    }
                )
        selected = max(candidates, key=lambda item: (item["iou"], item["f1"], item["precision"]))

        final_lookup = _fit_histogram(archive, calibration_clip, calibration_frames)
        evaluation_metrics, per_frame = _evaluate(
            archive,
            evaluation_clip,
            evaluation_frames,
            final_lookup,
            threshold=selected["threshold"],
            open_size=selected["open_size"],
            close_size=selected["close_size"],
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        probe_indices = sorted({0, len(evaluation_frames) // 2, len(evaluation_frames) - 1})
        overlays: list[str] = []
        for index in probe_indices:
            filename = evaluation_frames[index]
            rgb, target = archive.pair(evaluation_clip, filename)
            prediction = _predict(
                rgb,
                final_lookup,
                selected["threshold"],
                selected["open_size"],
                selected["close_size"],
            )
            output_path = output_dir / f"overlay_{evaluation_clip}_{Path(filename).stem}.jpg"
            cv2.imwrite(str(output_path), _overlay(rgb, target, prediction))
            overlays.append(str(output_path))

    report = {
        "schema_version": 1,
        "algorithm": "balanced_hsv_histogram_likelihood",
        "calibration": {
            "clip": calibration_clip,
            "fit_frame_count": len(fit_frames),
            "validation_frame_count": len(validation_frames),
            "selected": selected,
        },
        "evaluation": {"clip": evaluation_clip, **evaluation_metrics},
        "per_frame": per_frame,
        "overlays": overlays,
        "status": "ok" if math.isfinite(evaluation_metrics["iou"]) else "failed",
        "evidence_boundary": "sample-domain semantic cable segmentation only; no endpoints, terminal identity, topology, fault type, or cabinet accuracy",
    }
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a lightweight MovingCables semantic segmentation baseline")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--calibration-clip", default="0003")
    parser.add_argument("--evaluation-clip", default="0006")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.archive, args.calibration_clip, args.evaluation_clip, args.output_dir)
    print(json.dumps({"status": report["status"], "calibration": report["calibration"], "evaluation": report["evaluation"]}, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
