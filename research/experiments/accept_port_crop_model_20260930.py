"""Frozen, two-phase source-image acceptance for the train01-only crop model.

Calibration reads only 24 inner-val normal train01 images. The validation phase
cannot run until calibration is saved with this script's hash and fixed rules.
No test01 access, labels in prediction selection, model training, or GUI edits.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

REPO = Path(r"E:\PythonProject10")
OUT = Path(__file__).resolve().parents[1] / "artifacts" / "port_crop_acceptance_20260930"
DATASET = REPO / "data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults/images"
MANIFEST = REPO / "data/derived/port_crop_training_20260929/dataset_manifest.json"
TRAIN_REPORT = REPO / "output/port_crop_training_fixed_20260929/full/report.json"
OLD_REPORT = REPO / "output/port_state_hints_validation_20260929/report.json"
WEIGHT = REPO / "output/port_crop_training_fixed_20260929/full/runs/rectports/weights/best.pt"
CONFIG = {
    "tile_size": 1280, "tile_stride": 960, "edge_margin": 16,
    "cross_tile_nms_iou": 0.5, "predict_imgsz": 960, "predict_conf_floor": 0.001,
    "predict_iou": 0.7, "predict_max_det": 300, "batch_tiles": 2,
    "minimum_hint_conf": 0.25, "normal_maximum_quantile": 0.95,
    "maximum_new_hints_per_image": 1,
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode {path}")
    return image


def transform_boxes(boxes: list[list[float]], matrix: np.ndarray) -> list[list[int]]:
    result = []
    for left, top, right, bottom in boxes:
        corners = cv2.perspectiveTransform(
            np.float32([[[left, top], [right, top], [right, bottom], [left, bottom]]]), matrix
        ).reshape(-1, 2)
        result.append([
            int(np.floor(corners[:, 0].min())), int(np.floor(corners[:, 1].min())),
            int(np.ceil(corners[:, 0].max())), int(np.ceil(corners[:, 1].max())),
        ])
    return result


def prerequisites() -> tuple[dict, dict, dict, list[str]]:
    manifest, training, old = load(MANIFEST), load(TRAIN_REPORT), load(OLD_REPORT)
    assert manifest["status"] == "complete" and manifest["source_split"] == "train01"
    assert manifest["source_group_disjoint"] and manifest["train_crops"] == 469 and manifest["inner_val_crops"] == 576
    assert training["status"] == "complete" and training["completed_epochs"] == 6
    assert training["dataset_manifest_sha256"] == sha(MANIFEST)
    assert training["outer_validation_or_test_used"] is False
    assert training["nonzero_gradient_steps"] > 0 and training["finite_gradients"]
    assert training["weights"]["best.pt"]["sha256"] == sha(WEIGHT)
    for path, digest in training["fingerprints"].items():
        assert sha(Path(path)) == digest, f"Training source changed: {path}"
    assert old["status"] == "complete" and old["split"] == "val01" and len(old["cases"]) == 30
    assert old["baseline"]["iou_ge_05"] == 4 and old["with_hints"]["iou_ge_05"] == 5
    assert old["with_hints"]["overlap_fragments"] == 164
    assert old["with_hints"]["fault_regions"] == 71 and old["with_hints"]["normal_regions"] == 37
    val_records = [r for r in manifest["records"] if r["split"] == "val"]
    val_sources = {r["source_image"] for r in val_records}
    train_sources = {r["source_image"] for r in manifest["records"] if r["split"] == "train"}
    assert len(val_sources) == 48 and len(val_records) == 576 and len(train_sources) == 192
    assert not (val_sources & train_sources)
    normal_names = sorted(name for name in val_sources if name.startswith("normal_") and name.endswith(".JPG"))
    assert len(normal_names) == 24
    old_names = [r["image"] for r in old["cases"]]
    assert len(set(old_names)) == 30 and sum(name.startswith("normal_") for name in old_names) == 15
    assert sum(not name.startswith("normal_") for name in old_names) == 15
    return manifest, training, old, normal_names


def make_model():
    os.environ["YOLO_CONFIG_DIR"] = str(OUT / "ultralytics_config")
    os.environ["YOLO_OFFLINE"] = "True"
    os.environ["YOLO_AUTOINSTALL"] = "False"
    (OUT / "ultralytics_config").mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO
    torch.set_num_threads(4)
    torch.manual_seed(20260929)
    model = YOLO(str(WEIGHT))
    assert model.task == "segment" and dict(model.names) == {0: "unplugged_plug", 1: "unplugged_jack"}
    return model


def predict_source(model, split: str, name: str, expected_sha: str, helpers) -> dict:
    tile_windows, near_artificial_edge, merge_tiled_ports = helpers
    source = DATASET / split / name
    assert split in ("train01", "val01") and sha(source) == expected_sha
    image = read_image(source)
    height, width = image.shape[:2]
    windows = tile_windows(width, height, CONFIG["tile_size"], CONFIG["tile_stride"])
    assert len(windows) == 12
    kept, rejected, raw_count = [], 0, 0
    started = time.perf_counter()
    for offset in range(0, len(windows), CONFIG["batch_tiles"]):
        group = windows[offset:offset + CONFIG["batch_tiles"]]
        crops = [image[y:b, x:r] for x, y, r, b in group]
        outputs = model.predict(
            crops, imgsz=CONFIG["predict_imgsz"], conf=CONFIG["predict_conf_floor"],
            iou=CONFIG["predict_iou"], max_det=CONFIG["predict_max_det"],
            device="cpu", verbose=False, save=False,
        )
        assert len(outputs) == len(group)
        for tile_id, (output, window) in enumerate(zip(outputs, group), offset):
            x, y, _, _ = window
            for box in output.boxes:
                raw_count += 1
                local = list(map(float, box.xyxy[0].tolist()))
                if near_artificial_edge(local, window, width, height, CONFIG["edge_margin"]):
                    rejected += 1
                    continue
                l, t, r, b = local
                kept.append({
                    "box_xyxy": [l + x, t + y, r + x, b + y],
                    "confidence": float(box.conf.item()),
                    "class_id": int(box.cls.item()), "source_tile": tile_id,
                })
    merged = merge_tiled_ports(kept, CONFIG["cross_tile_nms_iou"])
    assert sha(source) == expected_sha
    return {
        "split": split, "image": name, "source_sha256": expected_sha,
        "source_shape": [height, width], "windows": [list(w) for w in windows],
        "raw_predictions": raw_count, "edge_rejected": rejected,
        "edge_kept_predictions": kept, "merged_predictions": merged,
        "elapsed_seconds": time.perf_counter() - started,
    }


def calibration(manifest, training, old, normal_names, model, helpers):
    (OUT / "cache").mkdir(parents=True)
    # The source SHA comes from the frozen train-only plan, not from val/test.
    plan_path = REPO / "output/port_training_data_audit_20260929/crop_plan.json"
    assert manifest["plan_sha256"] == sha(plan_path)
    plan = load(plan_path)
    assert plan["status"] == "complete" and plan["source_split"] == "train01"
    sha_by_name = {r["image"]: r["source_sha256"] for r in plan["manifest"]}
    assert set(normal_names).issubset(sha_by_name)
    samples = []
    for index, name in enumerate(normal_names, 1):
        record = predict_source(model, "train01", name, sha_by_name[name], helpers)
        save(OUT / "cache" / f"train01_{Path(name).stem}.json", record)
        samples.append({"image": name, "source_sha256": sha_by_name[name],
                        "maximum_confidence": max((p["confidence"] for p in record["merged_predictions"]), default=0.0),
                        "merged_count": len(record["merged_predictions"])})
        save(OUT / "progress.json", {"phase": "calibration", "completed": index, "total": 24})
        print(f"normal calibration {index}/24 | {name} | merged={samples[-1]['merged_count']}", flush=True)
    threshold = max(CONFIG["minimum_hint_conf"], float(np.quantile([s["maximum_confidence"] for s in samples], CONFIG["normal_maximum_quantile"])))
    result = {
        "status": "frozen", "source_split": "train01_inner_val_normals", "source_count": 24,
        "threshold": threshold, "config": CONFIG, "samples": samples,
        "fingerprints": {"script": sha(Path(__file__)), "training_report": sha(TRAIN_REPORT),
                         "dataset_manifest": sha(MANIFEST), "weight": sha(WEIGHT),
                         "old_validation_report": sha(OLD_REPORT), "plan": sha(plan_path),
                         "port_tiling": sha(REPO / "inspection_agent/port_tiling.py"),
                         "port_state_hint": sha(REPO / "inspection_agent/port_state_hint.py")},
        "outer_validation_read": False,
        "warning": "Inner validation selected the checkpoint and calibrates threshold; not independent field calibration.",
    }
    save(OUT / "calibration.json", result)
    save(OUT / "progress.json", {"phase": "calibration_complete", "threshold": threshold, "completed": 24, "total": 24})
    print(json.dumps({"phase": "calibration_complete", "threshold": threshold}, indent=2), flush=True)


def validation(training, old, model, helpers):
    frozen = load(OUT / "calibration.json")
    assert frozen["status"] == "frozen" and frozen["source_count"] == 24 and frozen["outer_validation_read"] is False
    assert frozen["config"] == CONFIG
    required = {"script": Path(__file__), "training_report": TRAIN_REPORT,
                "dataset_manifest": MANIFEST, "weight": WEIGHT,
                "old_validation_report": OLD_REPORT,
                "port_tiling": REPO / "inspection_agent/port_tiling.py",
                "port_state_hint": REPO / "inspection_agent/port_state_hint.py"}
    for key, path in required.items():
        assert sha(path) == frozen["fingerprints"][key], f"Changed after calibration: {key}"
    assert sha(REPO / "output/port_training_data_audit_20260929/crop_plan.json") == frozen["fingerprints"]["plan"]
    from inspection_agent.port_tiling import select_additional_tile_hint
    from tools.evaluate_port_state_hints import transform_boxes as old_transform
    from tools.evaluate_mendeley_local_refinement import iou
    assert old_transform([[1, 2, 3, 4]], np.eye(3)) == transform_boxes([[1, 2, 3, 4]], np.eye(3))
    identities = {(m["split"], m["image"]): m["source_sha256"] for m in old["source_manifest"]}
    reference = read_image(DATASET / "train01/normal_073.JPG")
    reference_sha = sha(DATASET / "train01/normal_073.JPG")
    previous_tiling = load(REPO / "output/port_tiling_validation_20260929/report.json")
    assert previous_tiling["status"] == "complete"
    assert previous_tiling["fingerprints"][str(DATASET / "train01/normal_073.JPG")] == reference_sha
    cases = []
    for index, prior in enumerate(sorted(old["cases"], key=lambda c: c["image"]), 1):
        name = prior["image"]
        source = predict_source(model, "val01", name, identities["val01", name], helpers)
        save(OUT / "cache" / f"val01_{Path(name).stem}.json", source)
        matrix = np.asarray(prior["actual_homography"], dtype=np.float64)
        assert matrix.shape == (3, 3) and np.isfinite(matrix).all()
        assert prior["alignment"]["alignment_quality"]["reliable"]
        height, width = source["source_shape"]
        valid = cv2.warpPerspective(np.full((height, width), 255, np.uint8), matrix,
                                    (reference.shape[1], reference.shape[0]), flags=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        valid = cv2.erode(valid, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))) > 0
        transformed = []
        merged = source["merged_predictions"]
        for raw, bounds in zip(merged, transform_boxes([r["box_xyxy"] for r in merged], matrix)):
            l, t, r, b = bounds
            crop = valid[max(0, t):min(valid.shape[0], b), max(0, l):min(valid.shape[1], r)]
            coverage = float(crop.sum() / max(1, (r - l) * (b - t))) if crop.size else 0.0
            transformed.append({**dict(zip(("left", "top", "right", "bottom"), bounds)),
                                "class_id": raw["class_id"], "confidence": raw["confidence"],
                                "valid_warp_fraction": coverage, "support_tiles": raw["support_tiles"]})
        selected = select_additional_tile_hint(prior["parents"], prior["hints"], transformed, frozen["threshold"])
        assert selected["parents"] == prior["parents"] and selected["existing_hints"] == prior["hints"]
        cases.append({"image": name, "kind": prior["kind"], "targets": copy.deepcopy(prior["targets"]),
                      "source_classes": copy.deepcopy(prior["source_classes"]),
                      "parents": selected["parents"], "existing_hints": selected["existing_hints"],
                      "tile_hints": selected["tile_hints"], "selection_audit": selected["selection_audit"],
                      "aligned_predictions": transformed, "source_sha256": source["source_sha256"]})
        save(OUT / "partial_cases.json", cases)
        save(OUT / "progress.json", {"phase": "outer_validation", "completed": index, "total": 30})
        print(f"outer validation {index}/30 | {name} | new_hints={len(selected['tile_hints'])}", flush=True)

    def boxes(case, add):
        return case["parents"] + [h["box"] for h in case["existing_hints"]] + ([h["box"] for h in case["tile_hints"]] if add else [])

    def summarize(add):
        fault_cases = [c for c in cases if c["targets"]]
        normal_cases = [c for c in cases if not c["targets"]]
        per = lambda c: [max((iou(b, t) for b in boxes(c, add)), default=0.0) for t in c["targets"]]
        values = [v for c in fault_cases for v in per(c)]
        return {"fault_images_with_overlap": sum(any(v > 0 for v in per(c)) for c in fault_cases),
                "source_fragments": len(values), "overlap_fragments": sum(v > 0 for v in values),
                "iou_ge_01": sum(v >= 0.1 for v in values), "iou_ge_05": sum(v >= 0.5 for v in values),
                "mean_best_iou": float(np.mean(values)),
                "fault_regions": sum(len(boxes(c, add)) for c in fault_cases),
                "normal_regions": sum(len(boxes(c, add)) for c in normal_cases),
                "normal_images_with_added_hints": sum(bool(c["tile_hints"]) for c in normal_cases) if add else 0}

    baseline, proposed = summarize(False), summarize(True)
    assert baseline["iou_ge_05"] == 5 and baseline["overlap_fragments"] == 164
    assert baseline["fault_regions"] == 71 and baseline["normal_regions"] == 37
    per_image = []
    for case in cases:
        before = [max((iou(b, t) for b in boxes(case, False)), default=0.0) for t in case["targets"]]
        after = [max((iou(b, t) for b in boxes(case, True)), default=0.0) for t in case["targets"]]
        per_image.append({"image": case["image"], "new_hints": len(case["tile_hints"]),
                          "extra_precise": sum(n >= 0.5 and o < 0.5 for o, n in zip(before, after)),
                          "extra_overlap": sum(n > 0 and o == 0 for o, n in zip(before, after))})
    accepted = proposed["iou_ge_05"] > baseline["iou_ge_05"] and proposed["normal_regions"] <= baseline["normal_regions"]
    result = {"status": "complete", "phase": "outer_validation", "split": "val01",
              "threshold": frozen["threshold"], "config": CONFIG,
              "baseline": baseline, "proposed": proposed, "per_image": per_image,
              "accepted_for_further_integration_review": accepted, "formal_detection_changed": False,
              "weight_sha256": sha(WEIGHT), "calibration_sha256": sha(OUT / "calibration.json"),
              "source_manifest": [{"image": c["image"], "sha256": c["source_sha256"]} for c in cases],
              "reference_sha256": reference_sha,
              "warning": "Same scene/camera and prior val exposure; source-fragment box metrics, not field accuracy, cable identity, or continuity. An accepted gate only merits further review."}
    save(OUT / "report.json", result)
    save(OUT / "progress.json", {"phase": "complete", "completed": 30, "total": 30, "accepted": accepted})
    print(json.dumps({k: result[k] for k in ("threshold", "baseline", "proposed", "accepted_for_further_integration_review")}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("calibration", "validation"))
    args = parser.parse_args()
    manifest, training, old, normal_names = prerequisites()
    sys.path.insert(0, str(REPO))
    from inspection_agent.port_tiling import tile_windows, near_artificial_edge, merge_tiled_ports
    helpers = tile_windows, near_artificial_edge, merge_tiled_ports
    if args.phase == "calibration":
        assert not OUT.exists(), f"Output exists: {OUT}"
    else:
        assert (OUT / "calibration.json").is_file() and not (OUT / "report.json").exists()
    model = make_model()
    if args.phase == "calibration":
        # make_model created only the config directory; calibration owns all further files.
        assert set(OUT.iterdir()) == {OUT / "ultralytics_config"}
        calibration(manifest, training, old, normal_names, model, helpers)
    else:
        validation(training, old, model, helpers)


if __name__ == "__main__":
    main()
