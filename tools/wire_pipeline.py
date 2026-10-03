"""Unicode-safe first-stage pipeline for machine wire inspection on Windows.

Commands:
  label   - click four rigid anchors on an image
  roi     - draw connector rectangles on the correct reference image
  inspect - register an inspection image and save full-resolution connector crops
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


DEFAULT_ANCHORS = ["top_left_reference", "top_right_reference", "bottom_right_reference", "bottom_left_reference"]
COLORS = [(86, 84, 255), (255, 190, 0), (60, 220, 80), (255, 130, 0)]


def read_image(path: str | Path) -> np.ndarray | None:
    """Open Chinese Windows file paths, which cv2.imread may reject."""
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR) if buffer.size else None


def write_image(path: str | Path, image: np.ndarray) -> None:
    target = Path(path)
    ok, buffer = cv2.imencode(target.suffix, image)
    if not ok:
        raise ValueError(f"Cannot encode image: {target}")
    buffer.tofile(str(target))


def load_anchor_points(path: Path, order: list[str]) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    anchors = {anchor["id"]: anchor for anchor in payload["anchors"]}
    missing = [anchor_id for anchor_id in order if anchor_id not in anchors]
    if missing:
        raise ValueError(f"Anchor file is missing: {', '.join(missing)}")
    return np.float32([[anchors[anchor_id]["x"], anchors[anchor_id]["y"]] for anchor_id in order])


def command_label(args: argparse.Namespace) -> None:
    image = read_image(args.image)
    if image is None:
        raise SystemExit(f"Cannot open image: {args.image}")
    order = [value.strip() for value in args.anchors.split(",") if value.strip()]
    if len(order) < 4:
        raise SystemExit("At least four anchors are needed.")
    height, width = image.shape[:2]
    scale = min(1.0, 1200 / max(width, height))
    view_size = (round(width * scale), round(height * scale))
    points: list[tuple[float, float]] = []
    window = "Wire pipeline: label anchors | click, R reset, S save, Q quit"

    def draw() -> None:
        view = cv2.resize(image, view_size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        for index, (x, y) in enumerate(points):
            sx, sy = round(x * scale), round(y * scale)
            color = COLORS[index % len(COLORS)]
            cv2.circle(view, (sx, sy), 8, color, -1)
            cv2.putText(view, f"{index + 1}. {order[index]}", (sx + 12, sy - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(view, f"{index + 1}. {order[index]}", (sx + 12, sy - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
        next_text = "all anchors set" if len(points) == len(order) else f"next: {order[len(points)]}"
        cv2.putText(view, next_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        cv2.imshow(window, view)

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < len(order):
            points.append((x / scale, y / scale))
            draw()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    draw()
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            points.clear()
            draw()
        if key == ord("s"):
            if len(points) != len(order):
                print("Set every anchor before saving.")
                continue
            payload = {
                "image": args.image.name,
                "image_width": width,
                "image_height": height,
                "anchors": [
                    {"id": name, "x": x, "y": y, "x_normalized": x / width, "y_normalized": y / height}
                    for name, (x, y) in zip(order, points)
                ],
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"Saved {args.output}")
            break
    cv2.destroyAllWindows()


def command_roi(args: argparse.Namespace) -> None:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    root = args.config.parent.parent
    image = read_image(root / config["reference_image"])
    if image is None:
        raise SystemExit("Cannot open configured reference image.")
    ids = [value.strip() for value in args.ids.split(",") if value.strip()]
    if not ids or len(set(ids)) != len(ids):
        raise SystemExit("Provide unique connector IDs, for example J1,J2,J3.")
    print("Drag one rectangle per connector in this order: " + ", ".join(ids))
    print("Press Enter or Space to finish; Esc cancels.")
    boxes = cv2.selectROIs("Wire pipeline: connector ROIs", image, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    if len(boxes) != len(ids):
        raise SystemExit(f"Expected {len(ids)} boxes, got {len(boxes)}. Config unchanged.")
    config["connector_rois"] = {
        connector_id: {"left": int(x), "top": int(y), "right": int(x + w), "bottom": int(y + h)}
        for connector_id, (x, y, w, h) in zip(ids, boxes)
    }
    args.config.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(ids)} connector ROIs to {args.config}")


def visual_difference(reference_crop: np.ndarray, inspected_crop: np.ndarray) -> float:
    reference_gray = cv2.cvtColor(reference_crop, cv2.COLOR_BGR2GRAY)
    inspected_gray = cv2.cvtColor(inspected_crop, cv2.COLOR_BGR2GRAY)
    reference_gray = cv2.normalize(cv2.GaussianBlur(reference_gray, (3, 3), 0), None, 0, 255, cv2.NORM_MINMAX)
    inspected_gray = cv2.normalize(cv2.GaussianBlur(inspected_gray, (3, 3), 0), None, 0, 255, cv2.NORM_MINMAX)
    return float(cv2.absdiff(reference_gray, inspected_gray).mean())


def command_inspect(args: argparse.Namespace) -> None:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    root = args.config.parent.parent
    reference = read_image(root / config["reference_image"])
    inspected = read_image(args.image)
    if reference is None or inspected is None:
        raise SystemExit("Cannot open reference or inspection image.")
    order = config.get("anchor_order", DEFAULT_ANCHORS)
    current = load_anchor_points(args.anchors, order)
    reference_points = load_anchor_points(root / config["reference_anchor_labels"], order)
    transform = cv2.getPerspectiveTransform(current, reference_points)
    ref_height, ref_width = reference.shape[:2]
    registered = cv2.warpPerspective(inspected, transform, (ref_width, ref_height))
    args.output.mkdir(parents=True, exist_ok=True)
    write_image(args.output / "registered.jpg", registered)
    result: dict[str, object] = {"source_image": str(args.image), "decision": "manual_review", "connectors": {}}
    connectors: dict[str, dict[str, int]] = config.get("connector_rois", {})
    for connector_id, roi in connectors.items():
        left, top, right, bottom = (int(roi[key]) for key in ("left", "top", "right", "bottom"))
        crop = registered[top:bottom, left:right]
        reference_crop = reference[top:bottom, left:right]
        if crop.size == 0 or reference_crop.size == 0:
            result["connectors"][connector_id] = {"status": "uncertain", "reason": "invalid ROI"}
            continue
        crop_path = args.output / f"{connector_id}.jpg"
        reference_path = args.output / f"{connector_id}.reference.jpg"
        write_image(crop_path, crop)
        write_image(reference_path, reference_crop)
        result["connectors"][connector_id] = {
            "status": "uncertain",
            "confidence": 0.0,
            "visual_difference_score": round(visual_difference(reference_crop, crop), 3),
            "crop": str(crop_path),
            "reference_crop": str(reference_path),
        }
    (args.output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Machine wire inspection pipeline.")
    commands = parser.add_subparsers(dest="command", required=True)
    label = commands.add_parser("label", help="Click stable machine anchors")
    label.add_argument("--image", type=Path, required=True)
    label.add_argument("--output", type=Path, required=True)
    label.add_argument("--anchors", default=",".join(DEFAULT_ANCHORS))
    label.set_defaults(handler=command_label)
    roi = commands.add_parser("roi", help="Draw connector regions on the reference image")
    roi.add_argument("--config", type=Path, required=True)
    roi.add_argument("--ids", required=True)
    roi.set_defaults(handler=command_roi)
    inspect = commands.add_parser("inspect", help="Register and crop a machine inspection image")
    inspect.add_argument("--config", type=Path, required=True)
    inspect.add_argument("--image", type=Path, required=True)
    inspect.add_argument("--anchors", type=Path, required=True)
    inspect.add_argument("--output", type=Path, required=True)
    inspect.set_defaults(handler=command_inspect)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
