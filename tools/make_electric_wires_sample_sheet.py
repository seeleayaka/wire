"""Create a side-by-side visual audit sheet for Electric Wires test pairs."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def _fit(image: Image.Image, width: int, height: int) -> Image.Image:
    result = image.convert("RGB").copy()
    result.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(result, ((width - result.width) // 2, (height - result.height) // 2))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", nargs="+", default=["c1_0", "c2_0", "c3_0", "c4_0"])
    parser.add_argument("--extension", default=".jpg")
    args = parser.parse_args()

    pairs = args.pairs
    tile_width, tile_height, label_height = 560, 315, 32
    sheet = Image.new("RGB", (tile_width * 2, (tile_height + label_height) * len(pairs)), "#eeeeee")
    draw = ImageDraw.Draw(sheet)
    for row, pair in enumerate(pairs):
        y = row * (tile_height + label_height)
        rgb = Image.open(args.input_dir / f"{pair}_rgb{args.extension}")
        mask = Image.open(args.input_dir / f"{pair}_mask{args.extension}")
        sheet.paste(_fit(rgb, tile_width, tile_height), (0, y + label_height))
        sheet.paste(_fit(mask, tile_width, tile_height), (tile_width, y + label_height))
        draw.text((12, y + 8), f"{pair} - original RGB", fill="black")
        draw.text((tile_width + 12, y + 8), f"{pair} - binary cable mask", fill="black")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output, quality=95)
    print(args.output)


if __name__ == "__main__":
    main()
