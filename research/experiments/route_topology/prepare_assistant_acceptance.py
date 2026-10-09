"""Fresh original-photo inspection aids, no SAM or invented human confirmation."""
import json
from pathlib import Path
from PIL import Image, ImageDraw
from core import image_binding, sha256

ROOT=Path(__file__).resolve().parents[2]
out=ROOT/'artifacts/assistant_bundle_acceptance_20261007'
out.mkdir(exist_ok=False)
source=ROOT/'artifacts/mendeley_confirmed_bundle_demo_20261006/preparation_report.json'
prep=json.loads(source.read_text(encoding='utf-8'))
sheet=Image.new('RGB',(1100,1120),'white');draw=ImageDraw.Draw(sheet)
rows=[]
for i,row in enumerate(prep['cases']):
    original=row['original_source'];binding=image_binding(original['path'])
    if any(binding[k]!=original[k] for k in binding):raise ValueError('original drift')
    box=row['crop_context']['crop_box_xyxy']
    with Image.open(original['path']) as im:
        rgb=im.convert('RGB');rgb.save(out/(row['id']+'_full.png'))
        crop=rgb.crop(box);crop.save(out/(row['id']+'_native.png'))
        crop=crop.resize((520,520))
    x=(i%2)*550;y=(i//2)*560
    draw.text((x+12,y+8),row['id']+' | fresh original RGB crop',fill='black')
    sheet.paste(crop,(x+12,y+28))
    rows.append({'id':row['id'],'source':original,'crop_box_xyxy':box,
                 'native_crop_sha256':sha256(out/(row['id']+'_native.png'))})
sheet.save(out/'fresh_original_contact_sheet.png')
(out/'inspection_inventory.json').write_text(json.dumps({'kind':'assistant_inspection_aids_only',
    'source_preparation_sha256':sha256(source),'cases':rows,'human_reviews':0,'SAM_launched':False},indent=2),encoding='utf-8')
print(out)
