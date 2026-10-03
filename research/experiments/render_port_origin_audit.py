"""Source-only ROI audit, separate from mask/end-point observations."""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/segment_port_audit_20261001"
OUT.mkdir(parents=True, exist_ok=True)
mapping = json.loads((ROOT / "artifacts/terminal_mapping_topology_20261001/cabinet_1_draft.json").read_text(encoding="utf-8"))
with Image.open(mapping["image_binding"]["image_path"]) as image:
    plain = image.convert("RGB").crop((305, 145, 365, 205)).resize((600, 600))
annotated = plain.copy()
draw = ImageDraw.Draw(annotated)
for port in mapping["ports"]:
    x1,y1,x2,y2 = port["bbox_xyxy"]
    draw.rectangle(((x1-305)*10,(y1-145)*10,(x2-305)*10,(y2-145)*10), outline="#ff5656", width=3)
sheet = Image.new("RGB", (1200, 650), "#121a26")
font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 20)
draw = ImageDraw.Draw(sheet)
draw.text((10, 10), "原始照片放大（无SAM结果）", font=font, fill="white")
draw.text((610, 10), "原孔位草稿框（保留旧坐标）", font=font, fill="white")
sheet.paste(plain,(0,50)); sheet.paste(annotated,(600,50))
sheet.save(OUT / "source_only_port_roi_audit.png")
