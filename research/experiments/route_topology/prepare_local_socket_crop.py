"""Prepare all five fixed source contexts, without importing/starting SAM."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from run_prompt_contrast import digest,save,verify
from bundle_runtime_pins import source_pins
from run_reference_color_paths_source import exact_regions
from local_socket_crop import context_box

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/local_socket_crop_source_20261008'


def main():
    base=ROOT/'artifacts/mendeley_cable_socket_controls_20261006'
    p=json.loads((base/'protocol.json').read_text(encoding='utf-8'));verify(p['pins'])
    before=source_pins();assert before==p['mainline_pins']
    prep=json.loads((base/'preparation_report.json').read_text(encoding='utf-8'))
    scope=json.loads(Path(p['confirmed_scope_path']).read_text(encoding='utf-8'))
    cpu=next(a for a in scope['anchors'] if a['id']=='FAN_CPU')
    OUT.mkdir(exist_ok=False);cases=[];pins=dict(p['pins'])
    for row in prep['cases']:
        source=row['original_source'];assert digest(source['path'])==source['image_sha256']
        pose=next(a for a in row['anchors'] if a['id']=='FAN_CPU')
        assert pose['localization_proposal_supported'] and all(pose['gates'].values())
        box=context_box(cpu['bbox_xyxy'],pose['inspection_to_reference_local'],source['image_size'])
        im=Image.open(source['path']).convert('RGB').crop(box);path=OUT/(row['id']+'_crop.png');im.save(path)
        regions=exact_regions((im.height,im.width),box,scope,row['anchors'])
        region=OUT/(row['id']+'_CPU_region.png');Image.fromarray(regions['FAN_CPU'].astype('uint8')*255).save(region)
        cases.append(dict(id=row['id'],source=dict(path=str(path),image_sha256=digest(path),image_size=list(im.size),coordinate_frame='source_image_pixels'),
            original_source=source,crop_box_xyxy=box,CPU_region_path=str(region),
            phenotype=row['phenotype'],anchors=row['anchors'],baseline_directory=str(base/row['id'])))
        pins[str(path)]=digest(path);pins[str(region)]=digest(region)
    files=[Path(__file__),Path(__file__).with_name('local_socket_crop.py'),Path(__file__).with_name('run_local_socket_crop.py'),
           Path(__file__).with_name('analyze_local_socket_crop.py'),base/'preparation_report.json',ROOT/'artifacts/LOCAL_SOCKET_CROP_PROTOCOL_20261008.md']
    pins.update({str(f):digest(f) for f in files});verify(pins);assert source_pins()==before
    save(OUT/'protocol.json',dict(cases=cases,pins=pins,mainline_pins=before,sam_source=p['sam_source'],checkpoint=p['checkpoint'],
        confirmed_scope_path=p['confirmed_scope_path'],prompt='cable',retrieval_threshold=.5,acceptance_threshold=.75,
        padding_reference_long_side_factor=1,no_geometry_prompt=True,no_demo_images=True,model_observer_count=1))
    print(json.dumps(dict(status='prepared',cases=[dict(id=c['id'],crop=c['crop_box_xyxy']) for c in cases])))


if __name__=='__main__':main()
