"""Audit the deliberately small external-resource bundle.

This tool checks provenance and file integrity only. It does not convert
Image2Net circuit diagrams or MovingCables clips into cabinet ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _inside(root: Path, relative: str) -> Path:
    root = root.resolve()
    target = (root / relative).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"resource path escapes bundle root: {relative}")
    return target


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audit_image2net(bundle_root: Path, entry: dict[str, Any]) -> dict[str, Any]:
    validation = _inside(bundle_root, entry["relative_path"])
    golden_dir = validation / "golden"
    image_dir = validation / "images"
    golden = {path.stem: path for path in golden_dir.glob("*.json")}
    images = {path.stem: path for path in image_dir.glob("*.png")}
    invalid_json: list[str] = []
    for stem, path in golden.items():
        try:
            value = _json_object(path)
            netlist = value.get("ckt_netlist")
            if not isinstance(netlist, list):
                invalid_json.append(stem)
        except (OSError, ValueError, json.JSONDecodeError):
            invalid_json.append(stem)
    expected = int(entry["expected_pair_count"])
    paired = set(golden).intersection(images)
    checks = {
        "validation_directory_exists": validation.is_dir(),
        "license_file_exists": _inside(bundle_root, entry["license_relative_path"]).is_file(),
        "golden_json_count_matches": len(golden) == expected,
        "image_count_matches": len(images) == expected,
        "paired_stem_count_matches": len(paired) == expected,
        "all_golden_json_structures_valid": not invalid_json,
    }
    return {
        "status": "ok" if all(checks.values()) else "failed",
        "checks": checks,
        "counts": {
            "golden_json": len(golden),
            "images": len(images),
            "paired_stems": len(paired),
        },
        "invalid_json_stems": invalid_json,
        "claim_boundary": entry["claim_boundary"],
    }


def _audit_movingcables(bundle_root: Path, entry: dict[str, Any]) -> dict[str, Any]:
    archive = _inside(bundle_root, entry["relative_path"])
    actual_hash = _sha256(archive) if archive.is_file() else None
    expected_hash = entry["sha256"].lower()
    rgb: set[str] = set()
    flow: set[str] = set()
    clips: set[str] = set()
    if archive.is_file():
        with tarfile.open(archive, "r") as package:
            for member in package.getmembers():
                parts = Path(member.name).parts
                if len(parts) < 6 or parts[2:4] not in {
                    ("test", "rgb_clips"),
                    ("test", "normal_flow_first_back"),
                }:
                    continue
                image_type, clip, filename = parts[3], parts[4], parts[5]
                key = f"{clip}/{filename}"
                clips.add(clip)
                if image_type == "rgb_clips":
                    rgb.add(key)
                elif image_type == "normal_flow_first_back":
                    flow.add(key)
    expected_clips = set(entry["expected_clips"])
    checks = {
        "archive_exists": archive.is_file(),
        "sha256_matches": actual_hash == expected_hash,
        "expected_clips_present": expected_clips.issubset(clips),
        "rgb_and_flow_frames_pair": bool(rgb) and rgb == flow,
    }
    return {
        "status": "ok" if all(checks.values()) else "failed",
        "checks": checks,
        "sha256": actual_hash,
        "counts": {
            "clips": len(clips),
            "rgb_frames": len(rgb),
            "normal_flow_frames": len(flow),
            "paired_frames": len(rgb.intersection(flow)),
        },
        "claim_boundary": entry["claim_boundary"],
    }


def audit(manifest_path: Path) -> dict[str, Any]:
    manifest = _json_object(manifest_path)
    bundle_root = manifest_path.resolve().parent
    reports: dict[str, Any] = {}
    for entry in manifest.get("resources", []):
        if entry["id"] == "image2net_validation":
            reports[entry["id"]] = _audit_image2net(bundle_root, entry)
        elif entry["id"] == "movingcables_sample":
            reports[entry["id"]] = _audit_movingcables(bundle_root, entry)
        else:
            reports[entry["id"]] = {"status": "skipped", "reason": "no auditor"}
    return {
        "schema_version": 1,
        "bundle_status": "ok" if reports and all(item["status"] == "ok" for item in reports.values()) else "failed",
        "resources": reports,
        "evidence_boundary": "resource integrity only; not visual-to-topology accuracy",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the WireMind minimal external-resource bundle")
    parser.add_argument("--manifest", type=Path, default=ROOT / "external_resources" / "manifest.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.manifest)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["bundle_status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
