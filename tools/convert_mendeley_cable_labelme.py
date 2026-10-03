"""Convert Labelme cable polygons into one-class YOLO-seg labels."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def polygon_to_yolo(points: list[list[float]], width: int, height: int) -> str:
    if len(points) < 3:
        raise ValueError("polygon needs at least three points")
    values: list[str] = ["0"]
    for x, y in points:
        values.extend((f"{min(1.0, max(0.0, x / width)):.6f}", f"{min(1.0, max(0.0, y / height)):.6f}"))
    return " ".join(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labelme-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing directory: {args.output}")
    args.output.mkdir(parents=True)
    from PIL import Image

    counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        source_dir = args.labelme_root / split
        if not source_dir.is_dir():
            continue
        image_dir = args.output / "images" / split
        label_dir = args.output / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for json_path in sorted(source_dir.glob("*.json")):
            data = json.loads(json_path.read_text(encoding="utf-8"))
            image_path = source_dir / str(data.get("imagePath", ""))
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            with Image.open(image_path) as image:
                width, height = image.size
            rows: list[str] = []
            for shape in data.get("shapes", []):
                if shape.get("label") != "cable" or shape.get("shape_type") != "polygon":
                    continue
                rows.append(polygon_to_yolo(shape["points"], width, height))
            target_image = image_dir / image_path.name
            shutil.copy2(image_path, target_image)
            (label_dir / f"{image_path.stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
            counts[split] = counts.get(split, 0) + len(rows)
    yaml_path = args.output / "data.yaml"
    yaml_path.write_text(
        f"path: {args.output.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: cable\n",
        encoding="utf-8",
    )
    (args.output / "conversion_manifest.json").write_text(json.dumps({"class": "cable", "polygon_counts": counts}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "polygon_counts": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
