"""Deterministic scene contact sheets from existing ZIP, not model accuracy."""
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/terminal_scene_coverage_20261001';OUT.mkdir(parents=True,exist_ok=True)
ARCHIVE=Path('E:/PythonProject10/data/external_datasets/zenodo_terminal_strip_real_images.zip')
rows=[]
with zipfile.ZipFile(ARCHIVE) as archive:
    names=sorted(n for n in archive.namelist() if re.fullmatch(r'real_images/terminal_strip_\d{3}_[123]\.jpg',n))
    assert len(names)==300 and len(set(names))==300
    first=[n for n in names if n.endswith('_1.jpg')]
    assert len(first)==100
    # Fixed evenly spread other-view sample, no content-driven selection.
    groups=(1,12,23,34,45,56,67,78,89,100)
    other=[f'real_images/terminal_strip_{g:03d}_{v}.jpg' for g in groups for v in (2,3)]
    chosen=first+other
    for page in range(6):
        selected=chosen[page*20:(page+1)*20]
        sheet=Image.new('RGB',(1600,1040),'#f4f5f7');draw=ImageDraw.Draw(sheet)
        for i,name in enumerate(selected):
            raw=archive.read(name)
            with Image.open(io.BytesIO(raw)) as image:
                size=list(image.size);tile=image.convert('RGB');tile.thumbnail((395,226))
            x,y=(i%4)*400,(i//4)*208
            # Full-frame image retained; thumbnail must fit row height.
            tile.thumbnail((395,182))
            draw.text((x+3,y+3),Path(name).stem,fill='black');sheet.paste(tile,(x+3,y+23))
            rows.append(dict(member=name,image_sha256=hashlib.sha256(raw).hexdigest(),
                             image_size=size,contact_sheet=f'scenes_{page+1:02d}.png',review_kind='scene_thumbnail'))
        sheet.save(OUT/f'scenes_{page+1:02d}.png')
    # Native-resolution checks: fixed first/middle/last group, all views.
    native=[f'real_images/terminal_strip_{g:03d}_{v}.jpg' for g in (1,50,100) for v in (1,2,3)]
    for name in native:
        raw=archive.read(name)
        destination=OUT/Path(name).name
        if destination.exists():
            assert destination.read_bytes()==raw
        else:
            destination.write_bytes(raw)
        if not any(r['member']==name for r in rows):
            with Image.open(io.BytesIO(raw)) as image:size=list(image.size)
            rows.append(dict(member=name,image_sha256=hashlib.sha256(raw).hexdigest(),image_size=size,
                             contact_sheet=None,review_kind='native_resolution'))
    for group in (1,50,100):
        native_sheet=Image.new('RGB',(750,1560),'#f4f5f7');native_draw=ImageDraw.Draw(native_sheet)
        for view in (1,2,3):
            name=f'terminal_strip_{group:03d}_{view}.jpg'
            with Image.open(OUT/name) as image:region=image.convert('RGB').crop((550,300,1300,800))
            native_draw.text((3,(view-1)*520+3),name+' native crop [550,300,1300,800]',fill='black')
            native_sheet.paste(region,(0,(view-1)*520+20))
        native_sheet.save(OUT/f'native_group_{group:03d}.png')
    manifest=dict(archive_path=str(ARCHIVE),archive_sha256=hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
        archive_image_count=300,first_view_groups=100,other_view_thumbnail_count=20,
        native_review_members=native,selection='all first views + evenly spread ten groups views2/3 + native groups1/50/100 allviews',
        reviewed_unique_count=len(rows),images=rows,model_inference_used=False,
        full_archive_all_views_reviewed=False)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in manifest.items() if k!='images'},indent=2))
