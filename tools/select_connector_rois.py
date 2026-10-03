"""Draw connector regions on the reference image and save them to machine.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description="Select connector ROIs on a reference machine image.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ids", required=True, help="Comma-separated IDs in selection order, for example J1,J2,J3")
    args = parser.parse_args()

    connector_ids = [value.strip() for value in args.ids.split(",") if value.strip()]
    if not connector_ids or len(set(connector_ids)) != len(connector_ids):
        raise SystemExit("Provide one or more unique connector IDs.")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    root = args.config.parent.parent
    reference_path = root / config["reference_image"]
    image = cv2.imread(str(reference_path))
    if image is None:
        raise SystemExit(f"Cannot open reference image: {reference_path}")

    print("Draw a rectangle for each connector. Press Enter or Space when finished; Esc cancels.")
    print("Selection order: " + ", ".join(connector_ids))
    boxes = cv2.selectROIs("Select connector ROIs", image, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    if len(boxes) != len(connector_ids):
        raise SystemExit(f"Expected {len(connector_ids)} boxes, got {len(boxes)}. Config unchanged.")

    config["connector_rois"] = {
        connector_id: {"left": int(x), "top": int(y), "right": int(x + width), "bottom": int(y + height)}
        for connector_id, (x, y, width, height) in zip(connector_ids, boxes)
    }
    args.config.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(boxes)} connector ROIs to {args.config}")


if __name__ == "__main__":
    main()
