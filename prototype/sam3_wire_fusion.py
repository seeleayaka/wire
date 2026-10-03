"""SAM3-backed cable evidence for the cabinet DINO review.

SAM3 is deliberately run in its isolated CPU environment.  The GUI process
therefore remains independent from SAM3's heavy dependencies and can retain
the existing DINO output if SAM3 is unavailable or fails.  This module accepts
an already accepted, reference-coordinate inspection image; it never performs
its own registration or joins mask fragments into physical cables.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "cabinet_sam3_fusion_recipe.json"


class Sam3FusionError(RuntimeError):
    """SAM3-specific failure that permits the caller to retain DINO output."""


@dataclass(frozen=True)
class Sam3FusionSettings:
    python: Path
    site_packages: Path | None
    base_site_packages: Path | None
    source_root: Path
    runner: Path
    checkpoint: Path
    cache_dir: Path
    prompt: str
    confidence_threshold: float
    threads: int
    proximity_px: int
    green_min_sam_support_pixels: int
    yellow_min_component_pixels: int
    yellow_min_component_span_px: int
    yellow_min_dino_score_mean: float
    yellow_min_dino_score_p90: float

    def report(self) -> dict[str, Any]:
        return {
            "python": str(self.python),
            "site_packages": str(self.site_packages) if self.site_packages else None,
            "base_site_packages": str(self.base_site_packages) if self.base_site_packages else None,
            "source_root": str(self.source_root),
            "runner": str(self.runner),
            "checkpoint": str(self.checkpoint),
            "cache_dir": str(self.cache_dir),
            "prompt": self.prompt,
            "confidence_threshold": self.confidence_threshold,
            "threads": self.threads,
            "proximity_px": self.proximity_px,
            "green_min_sam_support_pixels": self.green_min_sam_support_pixels,
            "yellow_min_component_pixels": self.yellow_min_component_pixels,
            "yellow_min_component_span_px": self.yellow_min_component_span_px,
            "yellow_min_dino_score_mean": self.yellow_min_dino_score_mean,
            "yellow_min_dino_score_p90": self.yellow_min_dino_score_p90,
            "no_mask_joining": True,
        }


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _path_for_platform(value: Any, field: str, platform_key: str) -> Path:
    """Read a legacy string path or a per-platform executable mapping."""
    if isinstance(value, str):
        return _path(value)
    if not isinstance(value, dict):
        raise Sam3FusionError(f"{field} must be a path string or platform mapping")
    selected = value.get(platform_key, value.get("default"))
    if not isinstance(selected, str) or not selected:
        raise Sam3FusionError(f"{field} has no executable configured for {platform_key}")
    return _path(selected)


def _optional_path_for_platform(value: Any, field: str, platform_key: str) -> Path | None:
    if value is None:
        return None
    if isinstance(value, str) and not value:
        return None
    return _path_for_platform(value, field, platform_key)


def _runtime_platform_key() -> str:
    if os.name == "nt":
        return "windows"
    if sys.platform.startswith("linux"):
        return "linux"
    return sys.platform


def load_settings(config_path: Path = DEFAULT_CONFIG) -> Sam3FusionSettings:
    if not config_path.is_file():
        raise Sam3FusionError(f"SAM3 fusion configuration is missing: {config_path}")
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not raw.get("enabled", False):
        raise Sam3FusionError("SAM3 fusion is disabled in its cabinet recipe")
    return Sam3FusionSettings(
        python=_path_for_platform(raw["sam3_python"], "sam3_python", _runtime_platform_key()),
        site_packages=_optional_path_for_platform(raw.get("sam3_site_packages"), "sam3_site_packages", _runtime_platform_key()),
        base_site_packages=_optional_path_for_platform(raw.get("sam3_base_site_packages"), "sam3_base_site_packages", _runtime_platform_key()),
        source_root=_path(raw["sam3_source_root"]),
        runner=_path(raw["sam3_runner"]),
        checkpoint=_path(raw["checkpoint"]),
        cache_dir=_path(raw["cache_dir"]),
        prompt=str(raw["prompt"]),
        confidence_threshold=float(raw["confidence_threshold"]),
        threads=max(1, int(raw["threads"])),
        proximity_px=max(0, int(raw["proximity_px"])),
        green_min_sam_support_pixels=max(1, int(raw["green_min_sam_support_pixels"])),
        yellow_min_component_pixels=max(1, int(raw["yellow_min_component_pixels"])),
        yellow_min_component_span_px=max(1, int(raw["yellow_min_component_span_px"])),
        yellow_min_dino_score_mean=float(raw["yellow_min_dino_score_mean"]),
        yellow_min_dino_score_p90=float(raw["yellow_min_dino_score_p90"]),
    )


def validate_resources(settings: Sam3FusionSettings) -> None:
    missing = [
        path for path in (settings.python, settings.source_root / "sam3", settings.runner, settings.checkpoint)
        if not path.exists()
    ]
    if missing:
        raise Sam3FusionError("SAM3 resources are unavailable: " + ", ".join(str(path) for path in missing))


def _fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()[:20]


def _read_image(path: Path, flags: int) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)
    if image is None:
        raise Sam3FusionError(f"Cannot read image artifact: {path}")
    return image


def _write_image(path: Path, image: np.ndarray) -> None:
    ok, data = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise Sam3FusionError(f"Cannot encode artifact: {path}")
    data.tofile(str(path))


def _report_is_usable(output_dir: Path, source_path: Path, settings: Sam3FusionSettings) -> bool:
    report_path = output_dir / "report.json"
    union_path = output_dir / "mask_union.png"
    if not report_path.is_file() or not union_path.is_file():
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return (
        Path(str(report.get("input", ""))).resolve() == source_path.resolve()
        and report.get("prompt") == settings.prompt
        and float(report.get("confidence_threshold", -1.0)) == settings.confidence_threshold
    )


def _run_probe(source_path: Path, output_dir: Path, state_cache: Path, settings: Sam3FusionSettings) -> tuple[dict[str, Any], bool]:
    if _report_is_usable(output_dir, source_path, settings):
        return json.loads((output_dir / "report.json").read_text(encoding="utf-8")), True
    output_dir.mkdir(parents=True, exist_ok=True)
    state_cache.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(settings.python), "-B", str(settings.runner),
        "--input", str(source_path),
        "--checkpoint", str(settings.checkpoint),
        "--output-dir", str(output_dir),
        "--image-state-cache", str(state_cache),
        "--prompt", settings.prompt,
        "--threshold", str(settings.confidence_threshold),
        "--threads", str(settings.threads),
    ]
    environment = os.environ.copy()
    existing_python_path = environment.get("PYTHONPATH", "")
    python_paths = [str(settings.source_root)]
    if settings.site_packages is not None:
        python_paths.append(str(settings.site_packages))
    if settings.base_site_packages is not None:
        python_paths.append(str(settings.base_site_packages))
    if existing_python_path:
        python_paths.append(existing_python_path)
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=str(settings.source_root),
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout).strip().splitlines()
        tail = details[-1] if details else f"exit code {completed.returncode}"
        raise Sam3FusionError(f"SAM3 probe failed: {tail}")
    report_path = output_dir / "report.json"
    if not report_path.is_file() or not (output_dir / "mask_union.png").is_file():
        raise Sam3FusionError("SAM3 probe exited without its required mask artifacts")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["mainline_runner_seconds"] = round(time.perf_counter() - started, 3)
    return report, False


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    kernel_size = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def _component_records(mask: np.ndarray, direction: str, minimum_pixels: int, minimum_span: int) -> list[tuple[dict[str, Any], np.ndarray]]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    records: list[tuple[dict[str, Any], np.ndarray]] = []
    for label in range(1, count):
        left, top, width, height, pixels = (int(value) for value in stats[label])
        if pixels < minimum_pixels or max(width, height) < minimum_span:
            continue
        records.append((
            {
                "direction": direction,
                "bbox_xyxy": [left, top, left + width, top + height],
                "mask_pixels": pixels,
                "width": width,
                "height": height,
            },
            labels == label,
        ))
    return records


def _jet_to_score(heat: np.ndarray) -> np.ndarray:
    values = np.arange(256, dtype=np.uint8).reshape(256, 1)
    colors = cv2.applyColorMap(values, cv2.COLORMAP_JET).reshape(256, 3)
    lookup = np.zeros(1 << 24, dtype=np.uint8)
    color_codes = colors[:, 0].astype(np.uint32) | (colors[:, 1].astype(np.uint32) << 8) | (colors[:, 2].astype(np.uint32) << 16)
    lookup[color_codes] = np.arange(256, dtype=np.uint8)
    codes = heat[:, :, 0].astype(np.uint32) | (heat[:, :, 1].astype(np.uint32) << 8) | (heat[:, :, 2].astype(np.uint32) << 16)
    return lookup[codes]


def _overlap(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> bool:
    return max(first[0], second[0]) < min(first[2], second[2]) and max(first[1], second[1]) < min(first[3], second[3])


def _tier_dino_candidates(
    dino_candidates: list[dict[str, Any]],
    added: np.ndarray,
    missing: np.ndarray,
    minimum_sam_support_pixels: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep every valid DINO region for human review, tiered by SAM3 support."""
    height, width = added.shape
    green: list[dict[str, Any]] = []
    orange: list[dict[str, Any]] = []
    for source_index, candidate in enumerate(dino_candidates, start=1):
        left = max(0, min(width, int(candidate["left"])))
        top = max(0, min(height, int(candidate["top"])))
        right = max(0, min(width, int(candidate["right"])))
        bottom = max(0, min(height, int(candidate["bottom"])))
        if right <= left or bottom <= top:
            continue
        added_pixels = int(added[top:bottom, left:right].sum())
        missing_pixels = int(missing[top:bottom, left:right].sum())
        support = added_pixels + missing_pixels
        record = {
            "bbox_xyxy": [left, top, right, bottom],
            "source_dino_candidate_index": source_index,
            "sam_added_pixels": added_pixels,
            "sam_missing_pixels": missing_pixels,
            "sam_support_pixels": support,
            "dino_evidence_scores": candidate.get("evidence_scores", {}),
        }
        if support >= minimum_sam_support_pixels:
            green.append({
                "id": f"green_{len(green) + 1:02d}",
                "tier": "dino_and_sam",
                **record,
                "semantic": "visible_wire_related_change_region",
                "not_claimed": ["physical_cable_identity", "terminal_assignment", "electrical_continuity", "fault_type"],
            })
        else:
            orange.append({
                "id": f"orange_{len(orange) + 1:02d}",
                "tier": "dino_only_human_review",
                **record,
                "sam_support_state": "below_strong_support_threshold",
                "semantic": "visual_difference_candidate_without_strong_sam3_support",
                "not_claimed": ["confirmed_change", "physical_cable_identity", "terminal_assignment", "electrical_continuity", "fault_type"],
            })
    return green, orange


def _render_evidence(
    aligned: np.ndarray,
    added: np.ndarray,
    missing: np.ndarray,
    green: list[dict[str, Any]],
    orange: list[dict[str, Any]],
    yellow: list[dict[str, Any]],
) -> np.ndarray:
    drawing = aligned.astype(np.float32).copy()
    drawing[added] = drawing[added] * 0.62 + np.array([0, 80, 255]) * 0.38
    drawing[missing] = drawing[missing] * 0.62 + np.array([255, 160, 0]) * 0.38
    drawing = np.clip(drawing, 0, 255).astype(np.uint8)
    for record in green:
        left, top, right, bottom = record["bbox_xyxy"]
        cv2.rectangle(drawing, (left, top), (right, bottom), (0, 220, 0), 2)
        cv2.putText(drawing, record["id"], (left, max(20, top - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 0), 2, cv2.LINE_AA)
    for record in orange:
        left, top, right, bottom = record["bbox_xyxy"]
        cv2.rectangle(drawing, (left, top), (right, bottom), (0, 165, 255), 2)
        cv2.putText(drawing, record["id"], (left, max(20, top - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 165, 255), 2, cv2.LINE_AA)
    for record in yellow:
        left, top, right, bottom = record["bbox_xyxy"]
        cv2.rectangle(drawing, (left, top), (right, bottom), (0, 215, 255), 2)
        cv2.putText(drawing, record["id"], (left, max(20, top - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 215, 255), 2, cv2.LINE_AA)
    legend_lines = (
        "green: DINO+SAM | orange: DINO-only review",
        "yellow: SAM-led review | red/+ added | blue/- missing",
    )
    for text, baseline in zip(legend_lines, (23, 43)):
        cv2.putText(drawing, text, (15, baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(drawing, text, (15, baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
    return drawing


def _render_sam_difference(
    aligned: np.ndarray,
    added: np.ndarray,
    missing: np.ndarray,
    components: list[dict[str, Any]],
) -> np.ndarray:
    """Show only connected differences originating from the two SAM3 masks."""
    drawing = aligned.astype(np.float32).copy()
    drawing[added] = drawing[added] * 0.62 + np.array([0, 80, 255]) * 0.38
    drawing[missing] = drawing[missing] * 0.62 + np.array([255, 160, 0]) * 0.38
    drawing = np.clip(drawing, 0, 255).astype(np.uint8)
    for index, record in enumerate(components, start=1):
        left, top, right, bottom = record["bbox_xyxy"]
        is_added = record["direction"] == "inspection_only"
        color = (0, 80, 255) if is_added else (255, 160, 0)
        label = f"SAM{'+' if is_added else '-'} {index:02d}"
        cv2.rectangle(drawing, (left, top), (right, bottom), color, 2)
        cv2.putText(drawing, label, (left, max(20, top - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)
    return drawing


def fuse_already_aligned(
    reference_path: Path,
    aligned_inspection_path: Path,
    valid_mask_path: Path,
    dino_candidates: list[dict[str, Any]],
    dino_heat_path: Path,
    output_dir: Path,
    settings: Sam3FusionSettings | None = None,
) -> dict[str, Any]:
    """Run the proven SAM3 evidence stage after the mainline alignment gate."""
    settings = settings or load_settings()
    validate_resources(settings)
    if not reference_path.is_file() or not aligned_inspection_path.is_file() or not valid_mask_path.is_file() or not dino_heat_path.is_file():
        raise Sam3FusionError("SAM3 fusion requires the accepted reference, aligned inspection, valid mask, and DINO heatmap")

    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    reference_key = _fingerprint(reference_path)
    inspection_key = _fingerprint(aligned_inspection_path)
    reference_dir = settings.cache_dir / "reference_masks" / reference_key
    reference_state = reference_dir / "image_state.pt"
    inspection_dir = output_dir / "sam3" / "inspection"
    inspection_state = settings.cache_dir / "inspection_states" / f"{inspection_key}.pt"

    reference_report, reference_cache_hit = _run_probe(reference_path, reference_dir, reference_state, settings)
    inspection_report, inspection_cache_hit = _run_probe(aligned_inspection_path, inspection_dir, inspection_state, settings)

    reference_union = _read_image(reference_dir / "mask_union.png", cv2.IMREAD_GRAYSCALE) > 0
    inspection_union = _read_image(inspection_dir / "mask_union.png", cv2.IMREAD_GRAYSCALE) > 0
    valid_mask = _read_image(valid_mask_path, cv2.IMREAD_GRAYSCALE) > 0
    aligned = _read_image(aligned_inspection_path, cv2.IMREAD_COLOR)
    heat = _read_image(dino_heat_path, cv2.IMREAD_COLOR)
    if reference_union.shape != inspection_union.shape or valid_mask.shape != reference_union.shape or aligned.shape[:2] != reference_union.shape or heat.shape[:2] != reference_union.shape:
        raise Sam3FusionError("SAM3 masks, valid coverage, aligned image, and DINO heatmap do not share a coordinate system")

    added = inspection_union & valid_mask & ~_dilate(reference_union, settings.proximity_px)
    missing = reference_union & valid_mask & ~_dilate(inspection_union, settings.proximity_px)
    green, orange = _tier_dino_candidates(
        dino_candidates, added, missing, settings.green_min_sam_support_pixels
    )
    green_bounds = [tuple(record["bbox_xyxy"]) for record in green]
    orange_bounds = [tuple(record["bbox_xyxy"]) for record in orange]

    score = _jet_to_score(heat)
    raw_components = _component_records(added, "inspection_only", settings.yellow_min_component_pixels, settings.yellow_min_component_span_px)
    raw_components += _component_records(missing, "reference_only", settings.yellow_min_component_pixels, settings.yellow_min_component_span_px)
    yellow: list[dict[str, Any]] = []
    component_audit: list[dict[str, Any]] = []
    for record, mask in raw_components:
        bounds = tuple(record["bbox_xyxy"])
        scores = score[mask]
        mean = float(scores.mean()) if scores.size else 0.0
        p90 = float(np.percentile(scores, 90)) if scores.size else 0.0
        overlaps_green = any(_overlap(bounds, item) for item in green_bounds)
        overlaps_orange = any(_overlap(bounds, item) for item in orange_bounds)
        audit = {
            **record,
            "dino_traditional_score_mean": round(mean, 2),
            "dino_traditional_score_p90": round(p90, 2),
            "overlaps_green_region": overlaps_green,
            "overlaps_orange_region": overlaps_orange,
            "overlaps_dino_review_region": overlaps_green or overlaps_orange,
        }
        component_audit.append(audit)
        if not (overlaps_green or overlaps_orange) and mean >= settings.yellow_min_dino_score_mean and p90 >= settings.yellow_min_dino_score_p90:
            yellow.append({
                "id": f"yellow_{len(yellow) + 1:02d}",
                "tier": "sam_led_human_review",
                **audit,
                "semantic": "sam_led_visible_wire_change_for_human_review",
                "not_claimed": ["confirmed_change", "physical_cable_identity", "terminal_assignment", "electrical_continuity", "fault_type"],
            })

    output_dir.mkdir(parents=True, exist_ok=True)
    evidence = _render_evidence(aligned, added, missing, green, orange, yellow)
    sam_components = sorted(
        component_audit,
        key=lambda item: (item["dino_traditional_score_p90"], item["mask_pixels"]),
        reverse=True,
    )
    sam_difference = _render_sam_difference(aligned, added, missing, sam_components)
    _write_image(output_dir / "sam3_fusion_boxes.jpg", evidence)
    _write_image(output_dir / "sam3_mask_evidence.jpg", evidence)
    _write_image(output_dir / "sam3_difference_boxes.jpg", sam_difference)
    _write_image(output_dir / "sam3_inspection_only_mask.png", added.astype(np.uint8) * 255)
    _write_image(output_dir / "sam3_reference_only_mask.png", missing.astype(np.uint8) * 255)
    return {
        "status": "ok",
        "purpose": "SAM3 evidence fusion after accepted mainline alignment; all output remains manual review.",
        "settings": settings.report(),
        "reference_sam3": {
            "output_dir": str(reference_dir),
            "instance_count": reference_report.get("instance_count"),
            "cache_hit": reference_cache_hit,
            "image_state_cache": reference_report.get("image_state_cache"),
        },
        "inspection_sam3": {
            "output_dir": str(inspection_dir),
            "instance_count": inspection_report.get("instance_count"),
            "cache_hit": inspection_cache_hit,
            "image_state_cache": inspection_report.get("image_state_cache"),
            "timings": inspection_report.get("timings", {}),
            "mainline_runner_seconds": inspection_report.get("mainline_runner_seconds"),
        },
        "mask_difference": {
            "inspection_only_pixels": int(added.sum()),
            "reference_only_pixels": int(missing.sum()),
            "valid_warp_pixels": int(valid_mask.sum()),
            "no_mask_joining": True,
        },
        "raw_dino_candidate_count": len(dino_candidates),
        "green_regions": green,
        "orange_review_regions": orange,
        "yellow_review_regions": yellow,
        "sam_component_audit": sam_components,
        "dino_only_review_region_count": len(orange),
        "suppressed_visual_only_dino_candidate_count": 0,
        "invalid_dino_candidate_count": len(dino_candidates) - len(green) - len(orange),
        "review_regions": green + orange + yellow,
        "artifacts": {
            "fusion_overlay": "sam3_fusion_boxes.jpg",
            "mask_evidence": "sam3_mask_evidence.jpg",
            "sam_difference_boxes": "sam3_difference_boxes.jpg",
            "inspection_sam_overlay": "sam3/inspection/overlay.jpg",
            "inspection_only_mask": "sam3_inspection_only_mask.png",
            "reference_only_mask": "sam3_reference_only_mask.png",
        },
    }
