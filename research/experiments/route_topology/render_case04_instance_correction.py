"""Show two actual separate native SAM instances; no mask merging."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/wire_appearance_reference_v3_20261008'
prior=json.loads((ROOT/'artifacts/endpoint_pair_expanded30_v3_20261008/report.json').read_text(encoding='utf-8'))
row=next(r for r in prior['cases'] if r['id']=='case_04')
rgb=np.asarray(Image.open(row['source_path']).convert('RGB').crop(row['crop_box_xyxy']))
h,w=rgb.shape[:2]
canvas=Image.new('RGB',(w*2,h+55),'white');draw=ImageDraw.Draw(canvas)
for i,(identity,color,label) in enumerate([
    ('mask_003',[0,200,255],'other SAM instance; neither endpoint'),
    ('mask_004',[0,220,100],'selected instance; supports both endpoints')]):
    raw=np.asarray(Image.open(Path(row['native_run_directory'])/'sam'/(identity+'.png')).convert('L'))>0
    image=rgb.copy();image[raw]=(image[raw]*.4+np.array(color)*.6).astype('uint8')
    canvas.paste(Image.fromarray(image),(i*w,0))
    draw.text((i*w+8,h+8),identity,fill='black')
    draw.text((i*w+8,h+30),label,fill='black')
canvas.save(OUT/'case04_instances_separate.png')
