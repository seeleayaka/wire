"""Sensitivity of raw bundle support to global versus local anchor transforms.

Records disagreement explicitly. This does not override failed localization or
turn multiple mappings of the same raw mask into independent detector votes.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from bundle_support import inspect_support
from core import sha256
from run_paired_evidence import load_run
from run_review import save

ROOT=Path(__file__).resolve().parents[2]


def main():
    source=ROOT/'artifacts/mendeley_reference_geometry_prompt_20261005'
    protocol=json.loads((source/'protocol.json').read_text(encoding='utf-8'))
    case=next(c for c in protocol['cases'] if c['id']=='inspection')
    pose_path=ROOT/'artifacts/mendeley_local_anchor_pose_20261005/report.json'
    pose=json.loads(pose_path.read_text(encoding='utf-8'))
    for path,digest in pose['pins'].items():
        if sha256(path)!=digest:raise ValueError('pose evidence source drift')
    scope_path=ROOT/'artifacts/mendeley_reference_lead_calibration_20261005/reference_scope_draft.json'
    scope=json.loads(scope_path.read_text(encoding='utf-8'));binding=scope['reference_binding']
    reg_path=ROOT/'artifacts/mendeley_visible_fan_scope_20261005/registration/report.json'
    h=json.loads(reg_path.read_text(encoding='utf-8'))['registration']['source_to_reference_homography']
    mappings=[{'name':'global','mapping':h,'pose_gate_passed':True}]+[
        {'name':'local_'+a['id'],'mapping':a['inspection_to_reference_local'],
         'pose_gate_passed':a['reliable_localization']} for a in pose['anchors']]
    origin,records=load_run(source/'inspection/cable_plus_reference_anatomy_box')
    rows=[]
    for mapping in mappings:
        for record in records:
            if record['score']<.75 or record['boundary_truncated']:continue
            with Image.open(record['source_mask_path']) as opened:raw=np.asarray(opened.convert('L'))
            observation=inspect_support(raw,record['score'],record['record_id'],scope,binding,
                tuple(case['crop_box_xyxy'][:2]),mapping['mapping'])
            rows.append({'mapping_name':mapping['name'],'pose_gate_passed':mapping['pose_gate_passed'],
                'observation':observation})
    output=ROOT/'artifacts/mendeley_pose_support_sensitivity_20261005'
    if output.exists():raise FileExistsError('preserve previous sensitivity evidence')
    output.mkdir(exist_ok=False)
    result={'status':'complete','observations':rows,'model_observer_count':1,
        'failed_pose_gate_overridden':False,'new_confirmed_connections':0,
        'decision':'insufficient_evidence','deployed':False,'GT_read':False,
        'input_manifest_sha256':origin['manifest_sha256'],
        'source_pins':{str(p):sha256(p) for p in [Path(__file__),Path(__file__).with_name('bundle_support.py'),
            source/'protocol.json',pose_path,scope_path,reg_path]}}
    save(output/'report.json',result)
    print(json.dumps({'status':result['status'],'observations':[{
        'mapping':r['mapping_name'],'pose_gate_passed':r['pose_gate_passed'],
        'both_anchor_support':r['observation']['component_has_both_anchor_support'],
        'component_support':[c['anchor_pixel_support'] for c in r['observation']['components'] if c['both_anchor_support']]
        } for r in rows]},ensure_ascii=False))


if __name__=='__main__':main()
