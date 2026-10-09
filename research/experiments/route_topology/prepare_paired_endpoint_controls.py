"""Fixed two-box acquisition revision after photometric failure; source first."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from core import sha256,image_binding
from run_prompt_contrast import save,verify
from run_audit import source_pins
from paired_endpoint_prompts import endpoint_box

ROOT=Path(__file__).resolve().parents[2]

def main():
    src=ROOT/'artifacts/mendeley_cable_socket_controls_20261006'
    base=json.loads((src/'protocol.json').read_text(encoding='utf-8'));verify(base['pins'])
    before=source_pins();assert before==base['mainline_pins']
    scope=json.loads(Path(base['confirmed_scope_path']).read_text(encoding='utf-8'))
    prepared=json.loads((src/'preparation_report.json').read_text(encoding='utf-8'))['cases']
    out=ROOT/'artifacts/mendeley_paired_endpoint_controls_20261007';out.mkdir(exist_ok=False)
    files=[Path(__file__),Path(__file__).with_name('paired_endpoint_prompts.py'),Path(__file__).with_name('run_paired_endpoint_controls.py'),Path(__file__).with_name('audit_paired_endpoint_case.py'),src/'protocol.json',src/'report.json',
        ROOT/'artifacts/mendeley_contrast_cable_controls_20261007/report.json']
    pins={**base['pins'],**{str(p):sha256(p) for p in files}}
    cases=[]
    for row in prepared:
        origin=row['original_source'];assert sha256(origin['path'])==origin['image_sha256']
        crop=row['crop_context']['crop_box_xyxy'];rgb=np.asarray(Image.open(origin['path']).convert('RGB').crop(crop))
        path=out/(row['id']+'_original_crop.png');Image.fromarray(rgb).save(path);pins[str(path)]=sha256(path)
        boxes=[endpoint_box(a,next(p for p in row['anchors'] if p['id']==a['id']),crop) for a in scope['anchors']]
        cases.append(dict(row,crop_context=dict(row['crop_context'],source=dict(path=str(path),**image_binding(path))),
            positive_endpoint_boxes_cxcywh_normalized=boxes,verified_baseline_poses_and_appearance_reused=True))
    verify(pins);assert source_pins()==before
    save(out/'protocol.json',dict(cases=cases,pins=pins,mainline_pins=before,sam_source=base['sam_source'],checkpoint=base['checkpoint'],
        confirmed_scope_path=base['confirmed_scope_path'],baseline_report=str(src/'report.json'),
        acquisition_recipe='cable_plus_two_verified_endpoint_boxes',same_model_observer_count=1,
        prompt_sequence='cable then lead-box then socket-box, preserve only final native output; no mask unions',
        retrieval_threshold=.5,component_score_min=.75,native_connectivity=8,whole_mask_boundary_rejected=True,
        reference_gate_before_other_controls=True,all5_source_controls_retained=True,no_demo_images=True,
        source_gate='strict new visible gain, old supported losses0, exposed socket conflicts0',
        poses_and_appearance_explicitly_reused_for_single_acquisition_variable=True,
        no_electrical_claim=True,no_automatic_demo_extension=True,no_deployment=True))
    print('all five original source controls prepared; no clipping, no mask repair')

if __name__=='__main__':main()
