"""Source-only entry-region review; no endpoint or mask inputs."""
import json
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/source_entry_drafts_20261001'
OUT.mkdir(parents=True, exist_ok=True)
for case in ('cabinet_1', 'cabinet_2'):
    mapping = json.loads((ROOT / f'artifacts/terminal_mapping_topology_20261001/{case}_draft.json').read_text(encoding='utf-8'))
    with Image.open(mapping['image_binding']['image_path']) as image:
        box = (305, 145, 385, 225) if case == 'cabinet_1' else (287, 0, 575, 258)
        crop = image.convert('RGB').crop(box)
        scale = 10 if case == 'cabinet_1' else 3
        crop = crop.resize((crop.width*scale, crop.height*scale), Image.Resampling.NEAREST)
        draw = ImageDraw.Draw(crop)
        for x in range((box[0]//10+1)*10, box[2], 10):
            draw.line(((x-box[0])*scale, 0, (x-box[0])*scale, crop.height), fill='#a0a0a0', width=1)
            draw.text(((x-box[0])*scale+2, 2), str(x), fill='red')
        for y in range((box[1]//10+1)*10, box[3], 10):
            draw.line((0, (y-box[1])*scale, crop.width, (y-box[1])*scale), fill='#a0a0a0', width=1)
            draw.text((2, (y-box[1])*scale+2), str(y), fill='red')
        crop.save(OUT / f'{case}_source_grid.png')
