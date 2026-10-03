"""Label-free adapter from a finished GUI review to optional port hints."""
from __future__ import annotations
import copy
import math
from numbers import Real
from pathlib import Path

from inspection_agent.optional_port_crop_review import optional_port_crop_review, read_image


def _parent_geometry(parent):
    """Adapt coordinates, never clip, reorder or discard candidate evidence."""
    keys = ("left", "top", "right", "bottom")
    if not isinstance(parent, dict):
        raise ValueError("invalid_parent_geometry")
    legacy = [parent[k] for k in keys] if all(k in parent for k in keys) else None
    if any(k in parent for k in keys) and legacy is None:
        raise ValueError("invalid_parent_geometry")
    sam = parent.get("bbox_xyxy")
    values = sam if sam is not None else legacy
    if (not isinstance(values, (list, tuple)) or len(values) != 4
            or any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) for v in values)
            or not (0 <= values[0] < values[2] and 0 <= values[1] < values[3])):
        raise ValueError("invalid_parent_geometry")
    if legacy is not None and sam is not None and legacy != list(sam):
        raise ValueError("invalid_parent_geometry")
    if legacy is not None and any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) for v in legacy):
        raise ValueError("invalid_parent_geometry")
    return dict(zip(keys, values))


def run_gui_port_review(report, *, project, enabled=False, scene="unknown", prediction_provider=None):
    context = {
        "parents": copy.deepcopy(report.get("review_regions", [])),
        "existing_hints": copy.deepcopy(report.get("existing_port_hints", [])),
        "source_to_reference_homography": report.get("alignment", {}).get("source_to_reference_homography"),
        "alignment_reliable": report.get("alignment_quality", {}).get("reliable") is True,
        "source_sha256": report.get("image_fingerprints", {}).get("source_sha256"),
        "reference_sha256": report.get("image_fingerprints", {}).get("reference_sha256"),
    }
    fallback = {"schema_version":1,"status":"disabled" if not enabled else "fallback",
                "decision":"possible_difference_manual_review","parents":context["parents"],
                "existing_hints":context["existing_hints"],"tile_hints":[],
                "automatic_fault_verdict":False,"fallback_reason":None}
    if not enabled:
        return fallback
    fingerprints = report.get("image_fingerprints", {})
    if (fingerprints.get("stable_during_visual_analysis") is not True
            or context["source_sha256"] is None or context["reference_sha256"] is None
            or context["source_to_reference_homography"] is None):
        fallback["fallback_reason"] = "visual_geometry_provenance_missing"
        return fallback
    if any(item.get("dino_alignment_input") == "local_ecc_corrected"
           for item in report.get("local_alignment", []) if isinstance(item, dict)):
        fallback["fallback_reason"] = "local_alignment_not_supported"
        return fallback
    original_parents = copy.deepcopy(context["parents"])
    try:
        context["parents"] = [{**parent, **_parent_geometry(parent)} for parent in original_parents]
    except (ValueError, TypeError):
        fallback["fallback_reason"] = "invalid_parent_geometry"
        return fallback
    result = optional_port_crop_review(report.get("inspection", ""),report.get("reference", ""),context,
        project=project,calibration=Path(project) / "config/port_crop_calibration_frozen_20260930.json",
        enabled=True,scene=scene,prediction_provider=prediction_provider)
    # Hints retain indices into the unchanged original candidate sequence.
    result["parents"] = original_parents
    return result


def render_port_overlay(aligned_path, result, output_path):
    import cv2
    image = read_image(aligned_path)
    for parent in result["parents"]:
        l,t,r,b = [round(v) for v in _parent_geometry(parent).values()]
        cv2.rectangle(image,(l,t),(r,b),(0,215,255),3)
    for index,hint in enumerate(result["tile_hints"],1):
        l,t,r,b = [round(hint["box"][k]) for k in ("left","top","right","bottom")]
        cv2.rectangle(image,(l,t),(r,b),(220,180,0),5)
        cv2.putText(image,f"Port hint {index}",(l,max(20,t-8)),cv2.FONT_HERSHEY_SIMPLEX,.8,(220,180,0),2)
    target=Path(output_path)
    target.parent.mkdir(parents=True,exist_ok=True)
    ok,encoded=cv2.imencode(target.suffix,image)
    if not ok:
        raise ValueError("Cannot encode port overlay")
    encoded.tofile(str(target))
    return target
