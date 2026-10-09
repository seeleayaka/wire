"""Read-only source inventory diagnosis; RGB color is not wire identity."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
from run_prompt_contrast import digest, save, verify
from run_review import verified_run
from run_reference_color_paths_source import exact_regions
from bundle_runtime_pins import source_pins

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/blue_anchor_inventory_diagnosis_20261008'


def main():
    base = ROOT / 'artifacts/mendeley_cable_socket_controls_20261006'
    protocol = json.loads((base / 'protocol.json').read_text(encoding='utf-8'))
    verify(protocol['pins'])
    before = source_pins()
    assert before == protocol['mainline_pins']
    prep = json.loads((base / 'preparation_report.json').read_text(encoding='utf-8'))
    scope = json.loads(Path(protocol['confirmed_scope_path']).read_text(encoding='utf-8'))
    OUT.mkdir(exist_ok=False)
    rows = []
    for row in prep['cases']:
        source = row['original_source']; crop = row['crop_context']['crop_box_xyxy']
        assert digest(source['path']) == source['image_sha256']
        rgb = np.asarray(Image.open(source['path']).convert('RGB').crop(crop))
        regions = exact_regions(rgb.shape[:2], crop, scope, row['anchors'])
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        blue = (hsv[:, :, 1] >= 64) & (hsv[:, :, 2] >= 32) & np.isin(hsv[:, :, 0].astype(int)//10, [9, 10, 11])
        inv = verified_run(base / row['id'] / 'cable_plus_reference_anatomy_box', row['crop_context']['source']['path'])
        masks = []; accepted = np.zeros(blue.shape, dtype=bool)
        for path, score in zip(inv['paths'], inv['scores']):
            raw = np.asarray(Image.open(path).convert('L')) > 0
            if float(score) >= .75: accepted |= raw
            masks.append(dict(record_id=path.stem, score=float(score), sha256=digest(path),
                blue_anchor_hits={k:int((blue & raw & r).sum()) for k,r in regions.items()}))
        rows.append(dict(id=row['id'], original_source=source, masks=masks,
            RGB_blue_anchor_hits={k:int((blue & r).sum()) for k,r in regions.items()},
            accepted_union_blue_anchor_hits={k:int((blue & accepted & r).sum()) for k,r in regions.items()}))
        if row['id'] == 'source_visible_01':
            panels = []
            for label, pixels, color in [('Original RGB',np.zeros(blue.shape,bool),(0,0,0)),
                  ('RGB blue (not identity)',blue,(0,255,255)),
                  ('Blue inside accepted SAM',blue & accepted,(255,0,255))]:
                display = rgb.copy(); display[pixels] = color
                im = Image.fromarray(display); d = ImageDraw.Draw(im)
                for k,r in regions.items():
                    yy,xx=np.nonzero(r)
                    d.rectangle((int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())),outline='lime',width=1)
                    d.text((int(xx.min()),max(0,int(yy.min())-12)),k,fill='lime')
                panel = Image.new('RGB',(rgb.shape[1],rgb.shape[0]+30),'white'); panel.paste(im,(0,30))
                ImageDraw.Draw(panel).text((5,8),label,fill='black'); panels.append(panel)
            canvas=Image.new('RGB',(sum(p.width for p in panels),panels[0].height),'white')
            x=0
            for p in panels:canvas.paste(p,(x,0)); x+=p.width
            canvas.save(OUT/'source_visible_01.png')
            r=regions['FAN_CPU']; yy,xx=np.nonzero(r)
            box=(max(0,int(xx.min())-12),max(0,int(yy.min())-12),min(rgb.shape[1],int(xx.max())+13),min(rgb.shape[0],int(yy.max())+13))
            Image.fromarray(rgb).crop(box).resize(((box[2]-box[0])*5,(box[3]-box[1])*5)).save(OUT/'source_visible_01_cpu_rgb.png')
    verify(protocol['pins']); assert source_pins()==before
    save(OUT/'report.json',dict(status='complete',cases=rows,mainline_unchanged=True,
        fresh_SAM_calls=0,new_confirmed_connections=0,RGB_is_not_identity=True,
        script_sha256=digest(Path(__file__))))
    print(json.dumps(rows,ensure_ascii=False))


if __name__ == '__main__':main()
