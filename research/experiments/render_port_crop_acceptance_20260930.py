"""Render six local diagnostic crops from cached acceptance boxes, no inference."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "artifacts/port_crop_acceptance_20260930"
DATA = Path(r"E:\PythonProject10\data\external_datasets\mendeley_electrical_wiring_faults\Predictive Maintenance for Electrical Wiring Faults\images\val01")
OLD = Path(r"E:\PythonProject10\output\port_state_hints_validation_20260929\report.json")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def outline(image, box, color, width):
    xyxy = [int(box[k]) for k in ("left", "top", "right", "bottom")] if isinstance(box, dict) else list(map(int, box))
    cv2.rectangle(image, tuple(xyxy[:2]), tuple(xyxy[2:]), color, width)


def main():
    cases = [c for c in load(ROOT / "partial_cases.json") if c["tile_hints"]]
    prior = {c["image"]: c for c in load(OLD)["cases"]}
    assert len(cases) == 6
    panels = []
    for case in cases:
        name = case["image"]
        source = cv2.imdecode(np.fromfile(str(DATA / name), dtype=np.uint8), cv2.IMREAD_COLOR)
        assert source is not None
        matrix = np.asarray(prior[name]["actual_homography"], dtype=np.float64)
        aligned = cv2.warpPerspective(source, matrix, (source.shape[1], source.shape[0]))
        hint = case["tile_hints"][0]["box"]
        cx = (hint["left"] + hint["right"]) / 2
        cy = (hint["top"] + hint["bottom"]) / 2
        radius = 360
        l = max(0, int(cx - radius)); t = max(0, int(cy - radius))
        r = min(aligned.shape[1], int(cx + radius)); b = min(aligned.shape[0], int(cy + radius))
        crop = aligned[t:b, l:r].copy()
        def shift(box):
            values = [box[k] for k in ("left", "top", "right", "bottom")] if isinstance(box, dict) else box
            return [int(values[0] - l), int(values[1] - t), int(values[2] - l), int(values[3] - t)]
        for parent in case["parents"]:
            outline(crop, shift(parent), (0,255,255), 2)
        for target in case["targets"]:
            outline(crop, shift(target), (0,0,255), 2)
        outline(crop, shift(hint), (255,0,255), 4)
        canvas = np.full((760,760,3), 35, dtype=np.uint8)
        canvas[35:35+crop.shape[0], 20:20+crop.shape[1]] = crop
        cv2.putText(canvas, name + " | purple=new, red=target, yellow=parent", (12,24),
                    cv2.FONT_HERSHEY_SIMPLEX, .52, (255,255,255), 1, cv2.LINE_AA)
        panels.append(canvas)
    sheet = np.vstack([np.hstack(panels[i:i+2]) for i in (0,2,4)])
    target = ROOT / "new_hint_six_case_contact_sheet.jpg"
    assert not target.exists()
    ok, blob = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert ok
    blob.tofile(str(target))
    print(target)


if __name__ == "__main__":
    main()
