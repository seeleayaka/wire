"""Click four stable chassis features and write a reusable anchor-label JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


ANCHOR_ORDER = [
    "top_left_reference",
    "top_right_reference",
    "bottom_right_reference",
    "bottom_left_reference",
]
COLORS = [(86, 84, 255), (255, 190, 0), (60, 220, 80), (255, 130, 0)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Label four rigid machine anchors with mouse clicks.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    original = cv2.imread(str(args.image))
    if original is None:
        raise SystemExit(f"Cannot open image: {args.image}")
    height, width = original.shape[:2]
    max_display = 1200
    scale = min(1.0, max_display / max(width, height))
    display_size = (round(width * scale), round(height * scale))
    points: list[tuple[float, float]] = []
    window = "Machine anchors | Click in order; R reset; S save; Q quit"

    def draw() -> None:
        canvas = cv2.resize(original, display_size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        for index, (x, y) in enumerate(points):
            sx, sy = round(x * scale), round(y * scale)
            cv2.circle(canvas, (sx, sy), 8, COLORS[index], -1)
            cv2.circle(canvas, (sx, sy), 10, (255, 255, 255), 2)
            cv2.putText(canvas, f"{index + 1}. {ANCHOR_ORDER[index]}", (sx + 12, sy - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(canvas, f"{index + 1}. {ANCHOR_ORDER[index]}", (sx + 12, sy - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS[index], 1)
        next_name = "all anchors set" if len(points) == 4 else f"next: {ANCHOR_ORDER[len(points)]}"
        cv2.putText(canvas, next_name, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow(window, canvas)

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
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
            if len(points) != 4:
                print("Set all four anchors before saving.")
                continue
            payload = {
                "image": args.image.name,
                "image_width": width,
                "image_height": height,
                "anchors": [
                    {"id": anchor_id, "x": x, "y": y, "x_normalized": x / width, "y_normalized": y / height}
                    for anchor_id, (x, y) in zip(ANCHOR_ORDER, points)
                ],
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"Saved {args.output}")
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
