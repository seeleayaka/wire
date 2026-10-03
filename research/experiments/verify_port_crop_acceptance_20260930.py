"""Model-free replay of the frozen calibration and source-image acceptance."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "artifacts/port_crop_acceptance_20260930"
REPO = Path(r"E:\PythonProject10")
DATA = REPO / "data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults/images"
sys.path.insert(0, str(REPO))
from inspection_agent.port_tiling import tile_windows, near_artificial_edge, merge_tiled_ports, select_additional_tile_hint
from tools.evaluate_mendeley_local_refinement import iou


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    frozen, report, cases = (load(ROOT / name) for name in ("calibration.json", "report.json", "partial_cases.json"))
    old = load(REPO / "output/port_state_hints_validation_20260929/report.json")
    dataset_manifest = load(REPO / "data/derived/port_crop_training_20260929/dataset_manifest.json")
    assert report["status"] == "complete" and frozen["status"] == "frozen"
    assert sha(ROOT / "calibration.json") == report["calibration_sha256"]
    assert sha(REPO / "output/port_crop_training_fixed_20260929/full/runs/rectports/weights/best.pt") == report["weight_sha256"]
    assert frozen["fingerprints"]["script"] == sha(Path(__file__).with_name("accept_port_crop_model_20260930.py"))
    assert frozen["fingerprints"]["old_validation_report"] == sha(REPO / "output/port_state_hints_validation_20260929/report.json")
    assert frozen["fingerprints"]["dataset_manifest"] == sha(REPO / "data/derived/port_crop_training_20260929/dataset_manifest.json")
    assert frozen["config"] == report["config"]
    val_sources = {r["source_image"] for r in dataset_manifest["records"] if r["split"] == "val"}
    train_sources = {r["source_image"] for r in dataset_manifest["records"] if r["split"] == "train"}
    assert not (val_sources & train_sources)
    assert {s["image"] for s in frozen["samples"]} == {n for n in val_sources if n.startswith("normal_")}
    maxima = []
    for sample in frozen["samples"]:
        name = sample["image"]
        cached = load(ROOT / "cache" / f"train01_{Path(name).stem}.json")
        assert cached["split"] == "train01" and cached["source_sha256"] == sample["source_sha256"] == sha(DATA / "train01" / name)
        assert cached["windows"] == [list(x) for x in tile_windows(cached["source_shape"][1], cached["source_shape"][0])]
        assert merge_tiled_ports(cached["edge_kept_predictions"]) == cached["merged_predictions"]
        maximum = max((r["confidence"] for r in cached["merged_predictions"]), default=0.0)
        assert maximum == sample["maximum_confidence"]
        maxima.append(maximum)
    threshold = max(0.25, float(np.quantile(maxima, 0.95)))
    assert threshold == report["threshold"] == frozen["threshold"]
    old_cases = {case["image"]: case for case in old["cases"]}
    assert len(cases) == len(old_cases) == 30
    reference_path = DATA / "train01/normal_073.JPG"
    assert sha(reference_path) == report["reference_sha256"]
    reference = cv2.imdecode(np.fromfile(str(reference_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert reference is not None
    values_old, values_new = [], []
    precise_added, class_matched_precise, unmatched, normal_added = 0, 0, 0, 0
    for case in cases:
        name = case["image"]
        assert name in old_cases and case["parents"] == old_cases[name]["parents"]
        assert case["existing_hints"] == old_cases[name]["hints"]
        cached = load(ROOT / "cache" / f"val01_{Path(name).stem}.json")
        assert cached["source_sha256"] == case["source_sha256"] == sha(DATA / "val01" / name)
        assert cached["windows"] == [list(x) for x in tile_windows(cached["source_shape"][1], cached["source_shape"][0])]
        assert cached["raw_predictions"] == len(cached["edge_kept_predictions"]) + cached["edge_rejected"]
        for prediction in cached["edge_kept_predictions"]:
            tile = cached["windows"][prediction["source_tile"]]
            x, y, _, _ = tile
            l, t, r, b = prediction["box_xyxy"]
            assert not near_artificial_edge([l-x, t-y, r-x, b-y], tile, cached["source_shape"][1], cached["source_shape"][0])
        assert merge_tiled_ports(cached["edge_kept_predictions"]) == cached["merged_predictions"]
        matrix = np.asarray(old_cases[name]["actual_homography"], dtype=np.float64)
        height, width = cached["source_shape"]
        valid = cv2.warpPerspective(np.full((height, width), 255, np.uint8), matrix,
                                    (reference.shape[1], reference.shape[0]), flags=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        valid = cv2.erode(valid, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))) > 0
        transformed = []
        for raw in cached["merged_predictions"]:
            l, t, r, b = raw["box_xyxy"]
            points = cv2.perspectiveTransform(np.float32([[[l,t],[r,t],[r,b],[l,b]]]), matrix).reshape(-1,2)
            L, T, R, B = int(np.floor(points[:,0].min())), int(np.floor(points[:,1].min())), int(np.ceil(points[:,0].max())), int(np.ceil(points[:,1].max()))
            crop = valid[max(0,T):min(valid.shape[0],B), max(0,L):min(valid.shape[1],R)]
            coverage = float(crop.sum()/max(1,(R-L)*(B-T))) if crop.size else 0.0
            transformed.append({"left":L,"top":T,"right":R,"bottom":B,"class_id":raw["class_id"],
                                "confidence":raw["confidence"],"valid_warp_fraction":coverage,
                                "support_tiles":raw["support_tiles"]})
        assert transformed == case["aligned_predictions"]
        selected = select_additional_tile_hint(case["parents"], case["existing_hints"], transformed, threshold)
        assert selected["tile_hints"] == case["tile_hints"] and selected["selection_audit"] == case["selection_audit"]
        assert len(case["tile_hints"]) <= 1
        if not case["targets"]:
            normal_added += len(case["tile_hints"])
        old_boxes = case["parents"] + [h["box"] for h in case["existing_hints"]]
        new_boxes = old_boxes + [h["box"] for h in case["tile_hints"]]
        for index, target in enumerate(case["targets"]):
            prior = max((iou(b,target) for b in old_boxes), default=0.0)
            updated = max((iou(b,target) for b in new_boxes), default=0.0)
            assert updated >= prior
            values_old.append(prior); values_new.append(updated)
            if prior < 0.5 <= updated:
                precise_added += 1
                if any(iou(h["box"],target) >= 0.5 and h["box"]["class_id"] + 3 == case["source_classes"][index] for h in case["tile_hints"]):
                    class_matched_precise += 1
        unmatched += sum(not any(iou(h["box"], t) >= 0.5 and h["box"]["class_id"] + 3 == cls
                                 for t,cls in zip(case["targets"],case["source_classes"])) for h in case["tile_hints"])
    assert len(values_old) == len(values_new) == 245
    assert sum(x >= 0.5 for x in values_old) == 5
    assert sum(x >= 0.5 for x in values_new) == 10
    assert precise_added == 5 and class_matched_precise == 5
    assert normal_added == 0 and unmatched == 1
    assert report["baseline"]["iou_ge_05"] == 5 and report["proposed"]["iou_ge_05"] == 10
    assert report["proposed"]["normal_regions"] == report["baseline"]["normal_regions"] == 37
    assert report["formal_detection_changed"] is False
    result = {"status":"verified", "source_images_rehashed":54, "cached_predictions_replayed":54,
              "frozen_threshold":threshold, "old_precise":5, "new_precise":10,
              "new_class_matched_precise":class_matched_precise,"new_unmatched_hints":unmatched,
              "new_normal_hints":normal_added,"formal_detection_changed":False,
              "warning":"Same-camera retrospective validation; not independent field accuracy."}
    target = ROOT / "replay_verification.json"
    assert not target.exists()
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
