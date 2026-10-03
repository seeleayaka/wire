"""Prepare a small blank Labelme addon containing normal Mendeley images."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--images", nargs="+", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing directory: {args.output}")
    args.output.mkdir(parents=True)
    manifest = []
    for name in args.images:
        source = args.source / name
        if not source.is_file():
            raise FileNotFoundError(source)
        target = args.output / source.name
        shutil.copy2(source, target)
        with Image.open(source) as image:
            width, height = image.size
        data = {
            "version": "5.8.1",
            "flags": {},
            "shapes": [],
            "imagePath": target.name,
            "imageData": None,
            "imageHeight": height,
            "imageWidth": width,
        }
        json_path = args.output / f"{source.stem}.json"
        json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest.append({"image": target.name, "json": json_path.name})
    (args.output / "README.txt").write_text(
        "Normal assembly cable-labeling addon.\n"
        "Draw only visible cable bodies; exclude labels, connectors, clips, drive bays, motherboard and chassis.\n"
        "Save each JSON in Labelme. These images are normal examples, not empty-background negatives.\n",
        encoding="utf-8",
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "images": len(manifest)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
