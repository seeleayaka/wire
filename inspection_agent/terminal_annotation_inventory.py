"""Import supplied object labels, never infer port identity or connection edges."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

TERMINAL_CLASSES = frozenset({
    "PT 6-QUATTRO", "UT 4-TWIN-MT", "PT 6", "UT 4", "UT 2,5-MT",
    "UT 4-QUATTRO", "PT 2,5", "PTS 2,5-QUATTRO", "PT 2,5-QUATTRO",
    "UT 2,5", "PT 4-MT", "PTS 4-TWIN", "PT 4-TWIN",
})
ACCESSORY_CLASSES = frozenset({"End Cover", "CLIPFIX 35"})
FIELDS = {"width", "height", "class", "xmin", "ymax", "xmax", "ymin"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_annotations(text: str, image_size: tuple[int, int]) -> dict:
    width, height = image_size
    if type(width) is not int or type(height) is not int or min(width, height) <= 0:
        raise ValueError("image dimensions must be positive integers")
    reader = csv.DictReader(text.lstrip("\ufeff").splitlines(), delimiter=";")
    if reader.fieldnames is None or len(reader.fieldnames) != len(FIELDS) or set(reader.fieldnames) != FIELDS:
        raise ValueError("unsupported annotation schema")
    records = []
    for row_number, row in enumerate(reader, 2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"row {row_number}: malformed columns")
        try:
            x1, y1, x2, y2 = (int(row[key]) for key in ("xmin", "ymin", "xmax", "ymax"))
            declared_size = (int(row["width"]), int(row["height"]))
        except (TypeError, ValueError) as error:
            raise ValueError(f"row {row_number}: invalid integer") from error
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"row {row_number}: box outside image or inverted")
        if declared_size != (x2 - x1, y2 - y1):
            raise ValueError(f"row {row_number}: declared object dimensions disagree")
        label = row["class"].strip()
        kind = "terminal_body" if label in TERMINAL_CLASSES else (
            "accessory" if label in ACCESSORY_CLASSES else (
                "bridge_object" if label == "Plug-in Bridge" else "unknown_object"))
        records.append({"source_row": row_number, "class": label, "kind": kind,
                        "bbox_xyxy": [x1, y1, x2, y2]})
    if not records:
        raise ValueError("annotations are empty")
    # Spatial IDs are image-local drafts, not printed labels or cross-image identities.
    records.sort(key=lambda item: (item["bbox_xyxy"][0], item["bbox_xyxy"][1], item["source_row"]))
    counts = {}
    for item in records:
        kind = item["kind"]
        counts[kind] = counts.get(kind, 0) + 1
        item["id"] = f"{kind}_{counts[kind]:03d}"
        item["evidence_origin"] = "supplied_object_annotation"
        item["identity_status"] = "unconfirmed"
        item["port_rois"] = []
    return {
        "schema_version": 1, "image_size": [width, height], "objects": records,
        "counts": counts, "expected_connections": None, "confirmed_connections": [],
        "topology_assessment": {"status": "insufficient_evidence",
            "reasons": ["port_identity_not_confirmed", "expected_connections_not_supplied",
                        "observed_connections_not_verified"]},
        "claim_boundary": "Imported body labels only; no detector accuracy, port assignment or electrical continuity claim.",
    }


def import_inventory(image_path: Path, csv_path: Path) -> dict:
    from PIL import Image
    with Image.open(image_path) as image:
        size = image.size
        image.verify()
    result = parse_annotations(csv_path.read_text(encoding="utf-8-sig"), size)
    result["source"] = {"image_path": str(image_path.resolve()), "annotation_path": str(csv_path.resolve()),
                        "image_sha256": sha256(image_path), "annotation_sha256": sha256(csv_path)}
    return result
