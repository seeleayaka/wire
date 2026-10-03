"""Cluster visibly empty-jack annotations into fixed-Dell ROI candidates.

The source class 4 locations are only candidates for manual naming.  This tool
does not treat proximity clusters as physical connector identities.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np


def read_source_boxes(path: Path) -> list[tuple[float, float, float, float]]:
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) != 5 or int(float(values[0])) != 4:
            continue
        _class_id, center_x, center_y, width, height = map(float, values)
        boxes.append((center_x, center_y, width, height))
    return boxes


def connected_components(points: list[dict[str, float]], radius: float) -> list[list[dict[str, float]]]:
    unvisited = set(range(len(points)))
    groups: list[list[dict[str, float]]] = []
    while unvisited:
        start = unvisited.pop()
        component = [start]
        queue = deque([start])
        while queue:
            index = queue.popleft()
            point = points[index]
            neighbours = [
                candidate
                for candidate in unvisited
                if (points[candidate]["cx"] - point["cx"]) ** 2 + (points[candidate]["cy"] - point["cy"]) ** 2 <= radius**2
            ]
            for candidate in neighbours:
                unvisited.remove(candidate)
                queue.append(candidate)
                component.append(candidate)
        groups.append([points[index] for index in component])
    return groups


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(path.suffix or ".jpg", image)
    if not ok:
        raise ValueError(f"cannot encode {path}")
    encoded.tofile(str(path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--radius", type=float, default=0.045)
    parser.add_argument("--padding", type=float, default=0.025)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    points: list[dict[str, float]] = []
    for label_path in sorted((args.dataset_root / "labels").glob("*/*.txt")):
        for center_x, center_y, width, height in read_source_boxes(label_path):
            points.append({"cx": center_x, "cy": center_y, "width": width, "height": height})
    clusters = connected_components(points, args.radius)
    records = []
    for items in sorted(clusters, key=len, reverse=True):
        left = max(0.0, min(item["cx"] - item["width"] / 2 for item in items) - args.padding)
        top = max(0.0, min(item["cy"] - item["height"] / 2 for item in items) - args.padding)
        right = min(1.0, max(item["cx"] + item["width"] / 2 for item in items) + args.padding)
        bottom = min(1.0, max(item["cy"] + item["height"] / 2 for item in items) + args.padding)
        records.append({"candidate_id": f"jack_roi_{len(records)+1:02d}", "annotations": len(items), "roi_normalized_xyxy": [round(left, 6), round(top, 6), round(right, 6), round(bottom, 6)]})
    args.output.mkdir(parents=True)
    image = read_image(args.reference)
    for index, record in enumerate(records):
        left, top, right, bottom = record["roi_normalized_xyxy"]
        x1, y1, x2, y2 = int(left * image.shape[1]), int(top * image.shape[0]), int(right * image.shape[1]), int(bottom * image.shape[0])
        color = tuple(int(value) for value in cv2.applyColorMap(np.array([[index * 37 % 255]], dtype=np.uint8), cv2.COLORMAP_HSV)[0, 0])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        cv2.putText(image, f"{record['candidate_id']} ({record['annotations']})", (x1, max(28, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    write_image(args.output / "jack_roi_candidates.jpg", image)
    report = {"purpose": "manual review of fixed-Dell empty-jack ROI candidates", "source_class": 4, "radius": args.radius, "padding": args.padding, "candidate_rois": records, "warning": "candidate IDs are geometric clusters only; name or reject them after visual review before training an ROI classifier."}
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"points": len(points), "clusters": len(records), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
