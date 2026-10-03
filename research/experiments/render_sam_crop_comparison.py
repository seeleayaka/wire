"""Paired first-view mask-coverage visualization, not an accuracy chart."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/sam_crop_coverage_20261001"
with Image.open(OUT / "cabinet_1_upper_right.png") as image:
    raw = image.convert("RGB")
with Image.open(ROOT / "artifacts/bound_sam_topology_20261001/cabinet_1_fresh/sam/mask_union.png") as image:
    baseline = np.asarray(image.convert("L"))[0:311, 310:620] > 0
with Image.open(OUT / "cabinet_1_crop_run/sam/mask_union.png") as image:
    cropped = np.asarray(image.convert("L")) > 0


def tint(mask):
    value = np.asarray(raw).astype(np.float32)
    value[mask] = value[mask]*.5 + np.array([40, 220, 180])*.5
    return Image.fromarray(value.astype(np.uint8))


font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 23)
small = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 18)
sheet = Image.new("RGB", (1860, 760), "#121a26")
draw = ImageDraw.Draw(sheet)
panels = [("原图固定右上裁块", raw), ("全幅SAM：同一位置", tint(baseline)), ("裁块SAM：同一位置", tint(cropped))]
for i, (title, panel) in enumerate(panels):
    draw.text((i*620+20, 15), title, font=font, fill="white")
    sheet.paste(panel.resize((620, 622)), (i*620, 55))
draw.text((20, 694), "提示 cable、阈值0.5不变；绿色是掩膜覆盖，不是已确认电缆或连接真值。", font=small, fill="white")
draw.text((20, 725), "原始机柜1同一视角对照；裁块边缘残段必须弃权，不据本图调阈值。", font=small, fill="white")
sheet.save(OUT / "first_view_comparison.png")
