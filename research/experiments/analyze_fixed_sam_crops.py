"""Replay fixed-crop geometry, translate it, then inspect coverage, not accuracy."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw
from sam_crop_geometry import translate_crop_record
from sam_endpoint_binding import file_sha256

PROJECT = Path("E:/PythonProject10")
sys.path.insert(0, str(PROJECT))
from inspection_agent.terminal_mapping import image_binding, review_mapped_topology
from inspection_agent.visible_segment_geometry import assess_visible_skeleton
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/sam_crop_coverage_20261001"
load = lambda path: json.loads(path.read_text(encoding="utf-8"))
protocol = load(OUT / "frozen_protocol.json")
assert protocol["status"] == "complete"
extractor = PROJECT / "manual_review/extract_sam3_visible_segment_endpoints.py"
spec = importlib.util.spec_from_file_location("fixed_crop_replay_extractor", extractor)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
summaries = []
for case in protocol["cases"]:
    run = Path(case["fresh_run_directory"])
    bound = load(run / "bound_endpoint_report.json")
    manifest = load(run / "run_manifest.json")
    sam = load(run / "sam/report.json")
    source_path = Path(case["source_binding"]["image_path"])
    assert image_binding(source_path) == case["source_binding"]
    assert bound["image_binding"] == case["crop_binding"]
    crop_input = Path(case["crop_binding"]["image_path"])
    assert image_binding(crop_input) == case["crop_binding"]
    with Image.open(source_path) as source_image, Image.open(crop_input) as crop_image:
        assert np.array_equal(np.asarray(source_image.crop(tuple(case["crop_box_xyxy"])).convert("RGB")),
                              np.asarray(crop_image.convert("RGB"))), "crop pixels do not match declared source rectangle"
    assert sam["image_state_cache_reused"] is False and manifest["status"] == "complete"
    for raw_path, sha in {**manifest["verified_files"], **manifest["runtime_fingerprints"]}.items():
        assert file_sha256(Path(raw_path)) == sha
    records = []
    replay = []
    source_union = np.zeros((case["source_binding"]["image_size"][1], case["source_binding"]["image_size"][0]), dtype=bool)
    x1, y1, x2, y2 = case["crop_box_xyxy"]
    for index in range(1, sam["instance_count"] + 1):
        with Image.open(run / "sam" / f"mask_{index:03d}.png") as image:
            binary = (np.asarray(image.convert("L")) > 0).astype(np.uint8)
        source_union[y1:y2, x1:x2] |= binary.astype(bool)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for cid in range(1, count):
            area = int(stats[cid, cv2.CC_STAT_AREA])
            if area < 50:
                continue
            component = labels == cid
            skeleton, _ = module.skeleton_endpoint_candidates(component)
            original = {"record_id": f"m{index:03d}_c{cid:02d}", "source_mask_id": index,
                "source_score": sam["scores"][index - 1], "component_id": cid,
                "component_pixels": area, **assess_visible_skeleton(skeleton)}
            replay.append(original)
            bx, by, bw, bh = [int(stats[cid, field]) for field in [cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT]]
            records.append(translate_crop_record(original, case["crop_box_xyxy"], case["source_binding"]["image_size"],
                [bx, by, bx+bw, by+bh], 2))
    assert replay == bound["records"]
    translated = deepcopy(bound)
    translated["image_binding"] = case["source_binding"]
    translated["records"] = records
    translated["coordinate_transform"] = {"type": "integer_crop_translation", "box_xyxy": case["crop_box_xyxy"],
        "crop_input_binding": case["crop_binding"], "source_binding": case["source_binding"], "pixels_scaled": False}
    (OUT / (case["case"] + "_source_endpoints.json")).write_text(json.dumps(translated, ensure_ascii=False, indent=2), encoding="utf-8")
    with Image.open(source_path) as image:
        original_rgb = image.convert("RGB")
    pixels = np.asarray(original_rgb).astype(np.float32)
    pixels[source_union] = .5*pixels[source_union] + .5*np.array([40, 220, 180])
    overlay = Image.fromarray(pixels.astype(np.uint8))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle(case["crop_box_xyxy"], outline="#ffb020", width=2)
    for record in records:
        for px, py in record["visible_ends_xy"]:
            draw.ellipse((px-3, py-3, px+3, py+3), fill="#ff6060")
    overlay.save(OUT / (case["case"] + "_source_overlay.png"))
    summary = {"case": case["case"], "sam_instances": sam["instance_count"], "components": len(records),
        "raw_two_end_geometry": sum(r["geometry_pair_eligible"] for r in replay),
        "crop_edge_abstentions": sum(r["crop_evidence"]["boundary_truncated"] for r in records),
        "retained_two_end_geometry": sum(r["geometry_pair_eligible"] for r in records),
        "mask_geometry_replay_exact": True, "crop_pixels_match_source_rectangle": True,
        "mask_pixels_in_quadrant": int(source_union.sum())}
    if case["case"] == "cabinet_1":
        mapping = load(ROOT / "artifacts/terminal_mapping_topology_20261001/cabinet_1_draft.json")
        assessment = review_mapped_topology(mapping, source_path, translated)
        assert assessment["decision"] == "insufficient_evidence" and assessment["observed_graph"] is None
        (OUT / "cabinet_1_topology_review.json").write_text(json.dumps(assessment, ensure_ascii=False, indent=2), encoding="utf-8")
        old_run = ROOT / "artifacts/bound_sam_topology_20261001/cabinet_1_fresh"
        with Image.open(old_run / "sam/mask_union.png") as image:
            baseline = np.asarray(image.convert("L")) > 0
        summary["whole_frame_mask_pixels_in_same_quadrant"] = int(baseline[y1:y2,x1:x2].sum())
        summary["draft_ports"] = []
        for port in mapping["ports"]:
            px1, py1, px2, py2 = port["bbox_xyxy"]
            contacts = [r["record_id"] for r in records if any(px1<=p[0]<=px2 and py1<=p[1]<=py2 for p in r["visible_ends_xy"])]
            summary["draft_ports"].append({"id": port["id"], "new_mask_pixels": int(source_union[py1:py2,px1:px2].sum()),
                "baseline_mask_pixels": int(baseline[py1:py2,px1:px2].sum()), "retained_endpoint_contacts": contacts,
                "identity_confirmed": False})
    summaries.append(summary)
(OUT / "coverage_report.json").write_text(json.dumps({"status": "verified", "cases": summaries,
    "not_accuracy_metrics": True, "no_automatic_connections": True, "second_view_has_no_paired_full_frame_baseline": True}, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summaries, ensure_ascii=False, indent=2))
