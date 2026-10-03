"""Bind newly generated endpoints to one audited, cache-free SAM input snapshot."""
from copy import deepcopy
import hashlib
import math
from pathlib import Path
from PIL import Image


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bind_current_run_endpoints(endpoint_report: dict, sam_report: dict,
                              snapshot: Path, manifest: dict) -> dict:
    """Legacy caches without this run manifest are deliberately unsupported."""
    if manifest.get("schema_version") != 1 or manifest.get("status") != "inference_inputs_and_outputs_verified":
        raise ValueError("requires a verified current-run manifest")
    binding = manifest.get("image_binding", {})
    if binding.get("coordinate_frame") != "source_image_pixels":
        raise ValueError("this producer only supports unwarped source pixels")
    if file_sha256(snapshot) != binding.get("image_sha256"):
        raise ValueError("snapshot identity changed")
    with Image.open(snapshot) as image:
        if list(image.size) != binding.get("image_size"):
            raise ValueError("snapshot dimensions disagree")
    if Path(sam_report.get("input", "")).resolve() != snapshot.resolve():
        raise ValueError("SAM report references a different snapshot")
    if sam_report.get("image_state_cache_reused") is not False:
        raise ValueError("unverified image-state cache cannot establish this run identity")
    files = manifest.get("verified_files")
    if not isinstance(files, dict) or not files:
        raise ValueError("verified artifact fingerprints missing")
    for raw_path, expected in files.items():
        if file_sha256(Path(raw_path)) != expected:
            raise ValueError("current-run artifact changed: " + raw_path)
    scores = sam_report.get("scores")
    count = sam_report.get("instance_count")
    if type(count) is not int or count < 0 or not isinstance(scores, list) or len(scores) != count:
        raise ValueError("SAM instance/score count mismatch")
    if any(isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1 for score in scores):
        raise ValueError("invalid SAM score")
    source_dir = Path(endpoint_report.get("source_dir", ""))
    if str((source_dir / "report.json").resolve()) not in files:
        raise ValueError("endpoint source report has no current-run fingerprint")
    # Bind the parsed report itself, not merely some file with matching path.
    import json
    actual_report = json.loads((source_dir / "report.json").read_text(encoding="utf-8"))
    if actual_report != sam_report:
        raise ValueError("parsed SAM report differs from verified source")
    with Image.open(source_dir / "input.jpg") as preview:
        if list(preview.size) != binding["image_size"]:
            raise ValueError("SAM preview coordinate size differs")
    if str((source_dir / "input.jpg").resolve()) not in files:
        raise ValueError("SAM preview fingerprint missing")
    mask_paths = sorted(p for p in source_dir.glob("mask_*.png") if p.stem.removeprefix("mask_").isdigit())
    if [p.name for p in mask_paths] != [f"mask_{i:03d}.png" for i in range(1, count + 1)]:
        raise ValueError("mask inventory mismatch")
    for mask_path in mask_paths:
        if str(mask_path.resolve()) not in files:
            raise ValueError("mask fingerprint missing")
        with Image.open(mask_path) as mask:
            if list(mask.size) != binding["image_size"]:
                raise ValueError("mask coordinate size differs")
    if endpoint_report.get("geometry_contract_version") != 1 or not isinstance(endpoint_report.get("records"), list):
        raise ValueError("geometry contract missing")
    for record in endpoint_report["records"]:
        index = record.get("source_mask_id")
        if type(index) is not int or not 1 <= index <= count or record.get("source_score") != scores[index - 1]:
            raise ValueError("endpoint mask identity or score differs from current SAM report")
    result = deepcopy(endpoint_report)
    result["image_binding"] = deepcopy(binding)
    result["provenance"] = {"origin": "fresh_local_sam3_inference", "run_id": manifest["run_id"],
        "image_state_cache_reused": False, "verified_files": deepcopy(files),
        "runtime_fingerprints": deepcopy(manifest.get("runtime_fingerprints", {}))}
    result["coverage_review"] = {"complete": False, "reviewer": None,
        "evidence_note": "Visible masks do not establish coverage of physical or hidden wiring."}
    return result
