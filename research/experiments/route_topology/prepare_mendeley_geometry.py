"""Freeze one reference-defined anatomy box, mapped automatically to inspection."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from core import image_binding
from prepare_mendeley_scope import inspection_scope
from run_prompt_contrast import digest, save, verify
from run_audit import source_pins

ROOT=Path(__file__).resolve().parents[2]


def main():
    previous=ROOT/'artifacts/mendeley_visible_fan_scope_20261005'
    old=json.loads((previous/'protocol.json').read_text(encoding='utf-8'))
    inference=json.loads((previous/'inference_report.json').read_text(encoding='utf-8'))
    if inference['status']!='complete':raise ValueError('existing SAM must finish first')
    verify(old['pins'])
    output=ROOT/'artifacts/mendeley_reference_geometry_prompt_20261005'
    if output.exists():raise FileExistsError('new experiment only')
    context=json.loads((previous/'context_protocol.json').read_text(encoding='utf-8'))['cases']
    registration=json.loads((previous/'registration/report.json').read_text(encoding='utf-8'))
    matrix=np.array(registration['registration']['source_to_reference_homography'])
    # New isolated recipe: an anatomy bounding box on REFERENCE ONLY. It is not
    # a mask endpoint/GT box, and inspection box is never manually adjusted.
    anatomy=[1420,900,1730,1230]
    output.mkdir(parents=True,exist_ok=False)
    cases=[]
    for side,entry in zip(['reference','inspection'],context):
        source=Path(entry['source_binding']['image_path'])
        binding=image_binding(source)
        if any(binding[k]!=entry['source_binding'][k] for k in binding):raise ValueError('source drift')
        crop_box=entry['crop_box_xyxy']
        with Image.open(source) as opened:crop=opened.convert('RGB').crop(crop_box)
        destination=output/(side+'_original_crop.png');crop.save(destination)
        box=anatomy if side=='reference' else inspection_scope(anatomy,matrix,binding['image_size'])
        local=[box[0]-crop_box[0],box[1]-crop_box[1],box[2]-crop_box[0],box[3]-crop_box[1]]
        if not (0<=local[0]<local[2]<=crop.width and 0<=local[1]<local[3]<=crop.height):
            raise ValueError('anatomy box outside exact crop; do not clip')
        normalized=[(local[0]+local[2])/(2*crop.width),(local[1]+local[3])/(2*crop.height),
                    (local[2]-local[0])/crop.width,(local[3]-local[1])/crop.height]
        cases.append({'id':side,'source':{'path':str(destination),**image_binding(destination)},
            'original_source':{'path':str(source),**binding},'crop_box_xyxy':crop_box,
            'positive_box_cxcywh_normalized':normalized,'positive_box_source_xyxy':box})
    files=[Path(__file__),Path(__file__).with_name('run_mendeley_geometry.py'),
        Path(__file__).with_name('run_prompt_contrast.py'),Path(__file__).with_name('core.py'),
        Path(__file__).with_name('prepare_mendeley_scope.py'),
        previous/'protocol.json',previous/'context_protocol.json',previous/'registration/report.json',
        *[Path(c['source']['path']) for c in cases],*[Path(c['original_source']['path']) for c in cases]]
    pins={**old['pins'],**{str(p.resolve()):digest(p) for p in files}}
    save(output/'protocol.json',{'schema_version':1,'cases':cases,'pins':pins,'mainline_pins':source_pins(),
        'sam_source':old['sam_source'],'checkpoint':old['checkpoint'],
        'recipes':['cable','cable_plus_reference_anatomy_box'],
        'retrieval_threshold':.5,'whole_mask_geometry_threshold':.75,
        'reference_anatomy_xyxy':anatomy,'fresh_image_encoder_required':True,
        'model_observer_count':1,'GT_read':False,'reference_review_confirmed':False,
        'evidence_target':'complete visible scoped mask only; not electrical continuity',
        'stop_if':'input/code drift, runtime failure or30min between-case budget; no threshold scan',
        'motivation':'preceding fixed-text experiment yielded0 complete two-anchor reference paths'})
    print(output/'protocol.json')


if __name__=='__main__':main()
