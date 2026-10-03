"""Replay geometry from the fresh run's masks; do not rerun inference or tune."""
import importlib.util
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw

PROJECT = Path("E:/PythonProject10")
sys.path.insert(0, str(PROJECT))
from inspection_agent.visible_segment_geometry import assess_visible_skeleton
from inspection_agent.terminal_mapping import image_binding, review_mapped_topology
from sam_endpoint_binding import file_sha256

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/bound_sam_topology_20261001/cabinet_1_fresh"
load = lambda p: json.loads(p.read_text(encoding="utf-8"))
manifest = load(RUN / "run_manifest.json")
assert manifest["status"] == "complete"
bound = load(RUN / "bound_endpoint_report.json")
sam = load(RUN / "sam/report.json")
assert sam["image_state_cache_reused"] is False
source = Path(manifest["image_binding"]["image_path"])
assert image_binding(source) == bound["image_binding"] == manifest["image_binding"]
assert file_sha256(Path(sam["input"])) == bound["image_binding"]["image_sha256"]
for raw, sha in dict(manifest["verified_files"], **manifest["runtime_fingerprints"]).items():
    assert file_sha256(Path(raw)) == sha, raw
extractor = PROJECT / "manual_review/extract_sam3_visible_segment_endpoints.py"
spec = importlib.util.spec_from_file_location("geometry_replay_extractor", extractor)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
expected = []
for mask_index in range(1, sam["instance_count"] + 1):
    with Image.open(RUN / "sam" / f"mask_{mask_index:03d}.png") as image:
        binary = (np.asarray(image.convert("L")) > 0).astype(np.uint8)
    components, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    for component in range(1, components):
        area = int(stats[component, cv2.CC_STAT_AREA])
        if area < 50:
            continue
        skeleton, _ = module.skeleton_endpoint_candidates(labels == component)
        expected.append({"record_id": f"m{mask_index:03d}_c{component:02d}", "source_mask_id": mask_index,
            "source_score": sam["scores"][mask_index - 1], "component_id": component,
            "component_pixels": area, **assess_visible_skeleton(skeleton)})
assert expected == bound["records"], "fresh geometry replay differs"
mapping = load(ROOT / "artifacts/terminal_mapping_topology_20261001/cabinet_1_draft.json")
review = review_mapped_topology(mapping, source, bound)
assert review == load(RUN / "topology_review.json")
assert review["decision"] == "insufficient_evidence" and review["observed_graph"] is None
hits = []
for port in mapping["ports"]:
    x1, y1, x2, y2 = port["bbox_xyxy"]
    records = [{"record_id": r["record_id"], "point": point, "score": r["source_score"]}
        for r in bound["records"] for point in r["visible_ends_xy"]
        if x1 <= point[0] <= x2 and y1 <= point[1] <= y2]
    hits.append({"port_draft_id": port["id"], "spatial_contacts_only": records, "port_confirmed": False})
with Image.open(RUN / "endpoints/endpoints_overlay.jpg") as image:
    overlay = image.convert("RGB")
draw = ImageDraw.Draw(overlay)
for port in mapping["ports"]:
    draw.rectangle(port["bbox_xyxy"], outline="#ffb020", width=2)
overlay.save(RUN / "draft_ports_and_fresh_endpoints.png")
report = {"status": "verified", "fresh_inference_no_cache": True, "record_count": len(expected),
    "eligible_count": sum(r["geometry_pair_eligible"] for r in expected),
    "runtime_and_artifact_hashes_match": True, "mask_geometry_replay_exact": True,
    "topology_review_replay_exact": True, "topology_decision": review["decision"],
    "draft_roi_contacts": hits, "field_accuracy_claimed": False,
    "not_claimed": ["confirmed terminal identity", "complete topology", "electrical continuity"]}
(RUN / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
