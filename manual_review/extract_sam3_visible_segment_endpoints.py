from __future__ import annotations

import argparse
import json
from math import ceil
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from inspection_agent.visible_segment_geometry import assess_visible_skeleton


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract visible-segment endpoints from existing SAM3 masks. "
            "Each mask and each disconnected component is handled independently."
        )
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-component-pixels", type=int, default=50)
    parser.add_argument(
        "--labels",
        action="store_true",
        help="Show component IDs beside endpoint dots in the full-image overview.",
    )
    return parser.parse_args()


def skeleton_endpoint_candidates(component: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    if hasattr(cv2, "ximgproc"):
        skeleton = cv2.ximgproc.thinning(component.astype(np.uint8) * 255)
    else:
        # The project venv has scikit-image but uses the non-contrib OpenCV build.
        from skimage.morphology import skeletonize

        skeleton = skeletonize(component).astype(np.uint8) * 255
    active = (skeleton > 0).astype(np.uint8)
    neighbours = cv2.filter2D(
        active,
        ddepth=cv2.CV_16S,
        kernel=np.ones((3, 3), dtype=np.int16),
        borderType=cv2.BORDER_CONSTANT,
    ) - active
    ys, xs = np.where((active > 0) & (neighbours == 1))
    return skeleton, [(int(x), int(y)) for x, y in zip(xs, ys)]


def select_visible_ends(candidates: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Compatibility helper: a branched tip set is never reduced to two tips."""
    return list(candidates) if len(candidates) == 2 else []


def draw_endpoint(
    draw: ImageDraw.ImageDraw,
    point: tuple[int, int],
    color: tuple[int, int, int],
    label: str | None,
) -> None:
    x, y = point
    radius = 7
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(0, 0, 0), width=2)
    if label:
        draw.text((x + 9, y - 8), label, fill=color, stroke_width=2, stroke_fill=(0, 0, 0))


def save_component_audit_sheet(
    base: Image.Image,
    items: list[tuple[str, np.ndarray]],
    output_path: Path,
) -> None:
    """Save one independent visible-mask component per tile for manual review."""
    columns = 6
    tile_width, tile_height = 280, 210
    image_width, image_height = base.size
    rows = max(1, ceil(len(items) / columns))
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), (244, 246, 248))
    draw = ImageDraw.Draw(sheet)

    for item_index, (record_id, component) in enumerate(items):
        ys, xs = np.where(component)
        left = max(0, int(xs.min()) - 20)
        top = max(0, int(ys.min()) - 20)
        right = min(image_width, int(xs.max()) + 21)
        bottom = min(image_height, int(ys.max()) + 21)
        crop = base.crop((left, top, right, bottom)).convert("RGBA")
        component_crop = component[top:bottom, left:right]
        tint = Image.new("RGBA", crop.size, (0, 220, 195, 0))
        tint.putalpha(Image.fromarray(component_crop.astype(np.uint8) * 145))
        crop = Image.alpha_composite(crop, tint).convert("RGB")
        crop.thumbnail((tile_width - 12, tile_height - 34), Image.Resampling.LANCZOS)

        column = item_index % columns
        row = item_index // columns
        x0 = column * tile_width
        y0 = row * tile_height
        draw.rectangle(
            (x0 + 2, y0 + 2, x0 + tile_width - 3, y0 + tile_height - 3),
            outline=(190, 196, 202),
            width=1,
        )
        draw.text((x0 + 8, y0 + 7), record_id, fill=(20, 35, 50))
        paste_x = x0 + (tile_width - crop.width) // 2
        paste_y = y0 + 28 + (tile_height - 32 - crop.height) // 2
        sheet.paste(crop, (paste_x, paste_y))

    sheet.save(output_path, quality=95)


def main() -> int:
    args = parse_args()
    input_path = args.source_dir / "input.jpg"
    report_path = args.source_dir / "report.json"
    mask_paths = sorted(
        path
        for path in args.source_dir.glob("mask_*.png")
        if path.stem.removeprefix("mask_").isdigit()
    )
    if not input_path.is_file() or not report_path.is_file() or not mask_paths:
        raise FileNotFoundError("source-dir must contain input.jpg, report.json, and mask_###.png files")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    base = Image.open(input_path).convert("RGB")
    overlay = base.copy()
    draw = ImageDraw.Draw(overlay)
    records: list[dict[str, object]] = []
    audit_items: list[tuple[str, np.ndarray]] = []

    for mask_index, mask_path in enumerate(mask_paths, start=1):
        binary = (np.asarray(Image.open(mask_path).convert("L")) > 0).astype(np.uint8)
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for component_id in range(1, component_count):
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            if area < args.min_component_pixels:
                continue
            component = labels == component_id
            skeleton, _ = skeleton_endpoint_candidates(component)
            geometry = assess_visible_skeleton(skeleton)
            selected_ends = geometry["visible_ends_xy"]
            record_id = f"m{mask_index:03d}_c{component_id:02d}"
            record = {
                "record_id": record_id,
                "source_mask_id": mask_index,
                "source_score": report["scores"][mask_index - 1],
                "component_id": component_id,
                "component_pixels": area,
                **geometry,
            }
            records.append(record)
            audit_items.append((record_id, component))
            for end_index, point in enumerate(selected_ends, start=1):
                color = (238, 86, 74) if end_index == 1 else (66, 133, 244)
                label = f"{record_id}:{end_index}" if args.labels else None
                draw_endpoint(draw, point, color, label)

    overlay.save(args.output_dir / "endpoints_overlay.jpg", quality=95)
    save_component_audit_sheet(base, audit_items, args.output_dir / "visible_segment_audit_sheet.jpg")
    (args.output_dir / "endpoint_report.json").write_text(
        json.dumps(
            {
                "source_dir": str(args.source_dir.resolve()),
                "geometry_contract_version": 1,
                "manual_confirmation_required": True,
                "source_prompt": report.get("prompt"),
                "source_confidence_threshold": report.get("confidence_threshold"),
                "min_component_pixels": args.min_component_pixels,
                "overview_labels": args.labels,
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    summary = {
        "visible_segment_components": len(records),
        "two_end_components": sum(len(record["visible_ends_xy"]) == 2 for record in records),
        "branch_or_fragment_components": sum(
            not record["geometry_pair_eligible"] for record in records
        ),
        "output_dir": str(args.output_dir.resolve()),
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
