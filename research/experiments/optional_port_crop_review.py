"""Default-off source-image port hint, frozen to the accepted PC scene.

The caller supplies source-to-reference geometry from a prior visual review.
On any failed prerequisite, return the original parents/hints without additions.
No annotation fields are consumed. Cached prediction mode is verifier-only.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

SCENE = "mendeley_pc_wiring_same_camera"
CALIBRATION_SHA = "bf8a034e398402de4e9c22ed13dedcbb410e717989f2202b1adc49712844e86d"
REFERENCE_SHA = "09f81923e44089603ccfb5989e370d07cc27f2e929ebfc70de945a08947b19ac"
CONFIG = {
    "tile_size": 1280, "tile_stride": 960, "edge_margin": 16,
    "cross_tile_nms_iou": 0.5, "predict_imgsz": 960, "predict_conf_floor": 0.001,
    "predict_iou": 0.7, "predict_max_det": 300, "batch_tiles": 2,
    "minimum_hint_conf": 0.25, "normal_maximum_quantile": 0.95,
    "maximum_new_hints_per_image": 1,
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class GateError(ValueError):
    pass


def require(condition, reason):
    if not condition:
        raise GateError(reason)


def aligned_predictions(merged, matrix, source_shape, reference_shape):
    """Exactly reproduce the frozen four-corner transform and valid coverage."""
    import cv2
    import numpy as np
    height, width = source_shape
    ref_height, ref_width = reference_shape
    valid = cv2.warpPerspective(np.full((height, width), 255, np.uint8), matrix,
                                (ref_width, ref_height), flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = cv2.erode(valid, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))) > 0
    result = []
    for raw in merged:
        l, t, r, b = raw["box_xyxy"]
        corners = cv2.perspectiveTransform(np.float32([[[l,t],[r,t],[r,b],[l,b]]]), matrix).reshape(-1, 2)
        require(np.isfinite(corners).all(), "nonfinite_transformed_box")
        L, T = int(np.floor(corners[:,0].min())), int(np.floor(corners[:,1].min()))
        R, B = int(np.ceil(corners[:,0].max())), int(np.ceil(corners[:,1].max()))
        crop = valid[max(0,T):min(ref_height,B), max(0,L):min(ref_width,R)]
        coverage = float(crop.sum()/max(1,(R-L)*(B-T))) if crop.size else 0.0
        result.append({"left":L,"top":T,"right":R,"bottom":B,
                       "class_id":raw["class_id"],"confidence":raw["confidence"],
                       "valid_warp_fraction":coverage,"support_tiles":raw["support_tiles"]})
    return result


def read_image(path):
    import cv2
    import numpy as np
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    require(image is not None, "image_decode_failed")
    return image


def predict(model, image):
    from inspection_agent.port_tiling import tile_windows, near_artificial_edge, merge_tiled_ports
    height, width = image.shape[:2]
    windows = tile_windows(width, height, CONFIG["tile_size"], CONFIG["tile_stride"])
    kept, rejected, raw_count = [], 0, 0
    for offset in range(0, len(windows), CONFIG["batch_tiles"]):
        group = windows[offset:offset + CONFIG["batch_tiles"]]
        outputs = model.predict([image[y:b,x:r] for x,y,r,b in group],
            imgsz=CONFIG["predict_imgsz"],conf=CONFIG["predict_conf_floor"],
            iou=CONFIG["predict_iou"],max_det=CONFIG["predict_max_det"],
            device="cpu",verbose=False,save=False)
        require(len(outputs) == len(group), "prediction_batch_mismatch")
        for tile_id, (output, window) in enumerate(zip(outputs, group), offset):
            x, y, _, _ = window
            for box in output.boxes:
                raw_count += 1
                local = list(map(float, box.xyxy[0].tolist()))
                if near_artificial_edge(local,window,width,height,CONFIG["edge_margin"]):
                    rejected += 1
                    continue
                l,t,r,b = local
                kept.append({"box_xyxy":[l+x,t+y,r+x,b+y],
                             "confidence":float(box.conf.item()),
                             "class_id":int(box.cls.item()),"source_tile":tile_id})
    return {"source_shape":[height,width], "windows":[list(w) for w in windows],
            "raw_predictions":raw_count,"edge_rejected":rejected,
            "edge_kept_predictions":kept,
            "merged_predictions":merge_tiled_ports(kept,CONFIG["cross_tile_nms_iou"])}


def optional_port_crop_review(source, reference, context, *, project, calibration,
                              enabled=False, scene="unknown", weight=None,
                              prediction_provider=None):
    """Return additive hints or unchanged baseline with an explicit fallback reason.

    prediction_provider is for isolated verification; it is not exposed by CLI.
    scene is a caller declaration, not an automatically recognized scene.
    """
    parents, existing = copy.deepcopy(context["parents"]), copy.deepcopy(context["existing_hints"])
    result = {"schema_version":1,"decision":"possible_difference_manual_review",
              "parents":parents,"existing_hints":existing,"tile_hints":[],
              "status":"disabled","fallback_reason":None,"automatic_fault_verdict":False,
              "scene_declared":scene,"scene_automatically_verified":False,
              "execution_mode":"verification_provider" if prediction_provider else "live_inference"}
    if not enabled:
        return result
    result["status"] = "fallback"
    try:
        import numpy as np
        root, source, reference = Path(project), Path(source), Path(reference)
        calibration = Path(calibration)
        weight = Path(weight) if weight else root / "output/port_crop_training_fixed_20260929/full/runs/rectports/weights/best.pt"
        require(scene == SCENE, "unsupported_scene")
        require(context.get("alignment_reliable") is True, "unreliable_alignment")
        require(sha(calibration) == CALIBRATION_SHA, "calibration_fingerprint_mismatch")
        frozen = load(calibration)
        require(frozen.get("status") == "frozen" and frozen.get("threshold") == .25
                and frozen.get("config") == CONFIG, "frozen_policy_mismatch")
        pins = frozen["fingerprints"]
        require(sha(weight) == pins["weight"], "weight_fingerprint_mismatch")
        for key, name in [("port_tiling","port_tiling.py"),("port_state_hint","port_state_hint.py")]:
            require(sha(root / "inspection_agent" / name) == pins[key], key + "_fingerprint_mismatch")
        source_sha = sha(source)
        require(source_sha == context.get("source_sha256"), "source_geometry_identity_mismatch")
        require(sha(reference) == REFERENCE_SHA and context.get("reference_sha256") == REFERENCE_SHA,
                "reference_fingerprint_mismatch")
        matrix = np.asarray(context.get("source_to_reference_homography"),dtype=np.float64)
        require(matrix.shape == (3,3) and np.isfinite(matrix).all()
                and np.linalg.matrix_rank(matrix) == 3, "invalid_homography")
        image, ref = read_image(source), read_image(reference)
        require(list(image.shape[:2]) == [2736,3648] and list(ref.shape[:2]) == [2736,3648],
                "unsupported_image_geometry")
        if prediction_provider:
            predictions = prediction_provider(image)
        else:
            import torch
            from ultralytics import YOLO
            torch.set_num_threads(4)
            torch.manual_seed(20260929)
            model = YOLO(str(weight))
            require(model.task == "segment" and dict(model.names) == {0:"unplugged_plug",1:"unplugged_jack"},
                    "model_class_contract_mismatch")
            predictions = predict(model,image)
        from inspection_agent.port_tiling import tile_windows, merge_tiled_ports, select_additional_tile_hint
        require(predictions["source_shape"] == list(image.shape[:2]), "prediction_shape_mismatch")
        require(predictions["windows"] == [list(w) for w in tile_windows(3648,2736)], "prediction_windows_mismatch")
        merged = merge_tiled_ports(predictions["edge_kept_predictions"],CONFIG["cross_tile_nms_iou"])
        require(merged == predictions["merged_predictions"], "prediction_merge_mismatch")
        transformed = aligned_predictions(merged,matrix,image.shape[:2],ref.shape[:2])
        selected = select_additional_tile_hint(parents,existing,transformed,.25)
        require(selected["parents"] == parents and selected["existing_hints"] == existing,
                "baseline_mutation_detected")
        require(len(selected["tile_hints"]) <= 1, "hint_budget_exceeded")
        require(sha(source) == source_sha and sha(reference) == REFERENCE_SHA and sha(weight) == pins["weight"]
                and sha(calibration) == CALIBRATION_SHA, "input_changed_during_prediction")
        result.update(selected)
        result.update(status="applied",fallback_reason=None,threshold=.25,
                      aligned_predictions=transformed,predictions=predictions,
                      source_sha256=source_sha,reference_sha256=REFERENCE_SHA,
                      weight_sha256=pins["weight"],calibration_sha256=CALIBRATION_SHA)
    except Exception as error:
        result["fallback_reason"] = str(error) if isinstance(error,GateError) else type(error).__name__ + ": " + str(error)
    return result
