"""Read verified source-pixel SAM runs; recompute paths; compare reviewed scope.

Run with --pair paired_ports.json --reference-run RUN --inspection-run RUN
--output NEW_DIRECTORY. Does not launch models or alter any existing result.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from core import (sha256, image_binding, text, reviewed, extract_mask,
                  validate_ports, assess_view, compare_views)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def verified_run(directory, image_path):
    directory = Path(directory).resolve()
    manifest = read(directory / "run_manifest.json")
    if manifest.get("schema_version") != 1 or manifest.get("status") != "complete":
        raise ValueError("requires a completed verified SAM run")
    binding = image_binding(image_path)
    if any(binding[k] != manifest.get("image_binding", {}).get(k) for k in binding):
        raise ValueError("SAM source and selected image identity/coordinate frame differ")
    if manifest.get("recipe", {}).get("no_registration") is not True:
        raise ValueError("only source-pixel SAM supported; no unverified warp reuse")
    files = manifest.get("verified_files")
    if not isinstance(files, dict) or not files:
        raise ValueError("SAM artifact fingerprints missing")
    # Only accept this run's own files; do not read arbitrary paths from a JSON.
    for raw, digest in files.items():
        path = Path(raw).resolve()
        if not path.is_relative_to(directory) or sha256(path) != digest:
            raise ValueError("SAM run artifact drift or path outside run")
    report_path = directory / "sam/report.json"
    if str(report_path) not in files:
        raise ValueError("SAM report not pinned")
    report = read(report_path)
    if report.get("image_state_cache_reused") is not False:
        raise ValueError("origin run image-state provenance unverified")
    snapshot = Path(report["input"]).resolve()
    if not snapshot.is_relative_to(directory) or image_binding(snapshot) != binding:
        raise ValueError("SAM input snapshot differs from source image")
    count, scores = report.get("instance_count"), report.get("scores")
    if type(count) is not int or count < 0 or not isinstance(scores, list) or len(scores) != count:
        raise ValueError("SAM inventory/score count mismatch")
    paths = sorted(p for p in (directory / "sam").glob("mask_*.png")
                   if p.stem.removeprefix("mask_").isdigit())
    if [p.name for p in paths] != [f"mask_{i:03}.png" for i in range(1, count + 1)]:
        raise ValueError("SAM mask inventory differs")
    for path in paths:
        if str(path.resolve()) not in files:
            raise ValueError("unfingerprinted mask")
        with Image.open(path) as mask:
            if list(mask.size) != binding["image_size"]:
                raise ValueError("SAM mask dimension/frame differs")
    return {"directory": str(directory), "manifest_sha256": sha256(directory / "run_manifest.json"),
            "image_binding": binding, "paths": paths, "scores": scores,
            "origin_evidence_kind": manifest.get("evidence_kind", "recorded_local_sam_run"),
            "recorded_runtime_fingerprints": manifest.get("runtime_fingerprints", {}),
            "reuse_existing_verified_masks": True, "fresh_inference_this_review": False}


def review_side(side, run_dir):
    if not isinstance(side, dict):
        raise ValueError("missing side of paired port map")
    image = Path(side["image_path"]).resolve()
    binding = image_binding(image)
    if side.get("image_binding") != binding:
        raise ValueError("annotation image hash/dimensions/frame mismatch")
    identities = validate_ports(side["ports"], binding["image_size"])
    selected = side.get("mask_ids")
    if not isinstance(selected, list) or any(type(i) is not int or i < 1 for i in selected) or len(set(selected)) != len(selected):
        raise ValueError("mask_ids must be unique positive integers")
    selection_ok = reviewed(side["selection_review"])
    text(side["scope_description"], "scope description")
    provenance = verified_run(run_dir, image)
    if any(i > len(provenance["paths"]) for i in selected):
        raise ValueError("selected mask ID absent in SAM run")
    records = []
    for i in selected:
        path = provenance["paths"][i - 1]
        with Image.open(path) as opened:
            pixels = np.asarray(opened.convert("L"))
        item = extract_mask(pixels, provenance["scores"][i - 1], path.stem)
        item.update(source_mask_sha256=sha256(path), source_mask_path=str(path))
        records.append(item)
    assessment = assess_view(records, side["ports"], binding["image_size"], side["coverage_review"])
    if not selection_ok:
        assessment["reasons"].append("selected_mask_semantics_or_scope_not_reviewed")
        assessment["complete_visible_scope"] = False
    assessment["scope_description"] = side["scope_description"]
    assessment["image_binding"] = binding
    # Path objects aren't JSON serializable. Provenance already captures hashes.
    provenance["mask_files"] = [{"path": str(p), "sha256": sha256(p)} for p in provenance.pop("paths")]
    assessment["provenance"] = provenance
    return assessment, identities, records


def render_overlay(side, records, destination):
    with Image.open(side["image_path"]) as opened:
        base = opened.convert("RGB")
    draw = ImageDraw.Draw(base)
    for port in side["ports"]:
        draw.rectangle(port["bbox_xyxy"], outline=(28, 111, 204), width=2)
        draw.text(tuple(port["bbox_xyxy"][:2]), port["id"], fill=(28, 111, 204))
    for record in records:
        g = record["geometry"]
        if g["path_xy"]:
            draw.line([tuple(p) for p in g["path_xy"]], fill=(0, 181, 133), width=2)
        for x, y in g["tips_xy"]:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(233, 113, 34))
    base.save(destination)


def execute(pair_path, reference_run, inspection_run, output):
    pair_path, output = Path(pair_path), Path(output)
    initial_pair_digest = sha256(pair_path)
    pair = read(pair_path)
    if type(pair.get("schema_version")) is not int or pair["schema_version"] != 1:
        raise ValueError("unsupported paired-map schema")
    text(pair["scene_type"], "scene_type")
    ref, ref_ids, ref_records = review_side(pair["reference"], reference_run)
    ins, ins_ids, ins_records = review_side(pair["inspection"], inspection_run)
    if ref_ids != ins_ids:
        raise ValueError("port identity universes differ across views")
    expected = pair.get("expected_connections")
    if expected is not None:
        for edge in expected:
            if edge.get("from") not in ref_ids or edge.get("to") not in ref_ids:
                raise ValueError("expected edge references undeclared port")
    result = compare_views(ref, ins, expected, pair["expected_review"])
    result.update(created_at=datetime.now(timezone.utc).isoformat(), pair_sha256=initial_pair_digest,
                  scene_type=pair["scene_type"], reference_run=str(Path(reference_run).resolve()),
                  inspection_run=str(Path(inspection_run).resolve()),
                  experiment_core_sha256=sha256(Path(__file__).with_name("core.py")),
                  runner_sha256=sha256(__file__),
                  expected_connections=expected, human_annotations=pair)
    # TOCTOU recheck before creating new output. No existing files overwritten.
    if sha256(pair_path) != initial_pair_digest:
        raise ValueError("annotations changed during review")
    for side, directory in ((pair["reference"], reference_run), (pair["inspection"], inspection_run)):
        rechecked = verified_run(directory, side["image_path"])
        previous = ref if side is pair["reference"] else ins
        if rechecked["manifest_sha256"] != previous["provenance"]["manifest_sha256"]:
            raise ValueError("origin SAM manifest changed during review")
    output.mkdir(parents=True, exist_ok=False)
    render_overlay(pair["reference"], ref_records, output / "reference_paths.png")
    render_overlay(pair["inspection"], ins_records, output / "inspection_paths.png")
    save(output / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--inspection-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = execute(args.pair, args.reference_run, args.inspection_run, args.output)
    print(json.dumps({"decision": result["decision"], "reasons": result["reasons"],
                      "fresh_inference": False, "output": str(args.output.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
