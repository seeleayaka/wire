"""All fresh cable+box masks: raw component/anchor support, no acceptance override."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from bundle_support import inspect_support
from core import sha256
from run_audit import source_pins
from run_paired_evidence import load_run
from run_prompt_contrast import verify
from run_review import save

ROOT=Path(__file__).resolve().parents[2]


def main():
    evidence=ROOT/'artifacts/mendeley_reference_geometry_prompt_20261005'
    protocol=json.loads((evidence/'protocol.json').read_text(encoding='utf-8'))
    verify(protocol['pins'])
    if source_pins()!=protocol['mainline_pins']:raise ValueError('mainline drift')
    scope_path=ROOT/'artifacts/mendeley_reference_lead_calibration_20261005/reference_scope_draft.json'
    scope=json.loads(scope_path.read_text(encoding='utf-8'));binding=scope['reference_binding']
    reg_path=ROOT/'artifacts/mendeley_visible_fan_scope_20261005/registration/report.json'
    registration=json.loads(reg_path.read_text(encoding='utf-8'))['registration']
    if not registration['alignment_quality']['reliable']:raise ValueError('registration failed')
    cases=[]
    for case in protocol['cases']:
        origin,records=load_run(evidence/case['id']/'cable_plus_reference_anatomy_box')
        matrix=None if case['id']=='reference' else registration['source_to_reference_homography']
        rows=[]
        for record in records:
            with Image.open(record['source_mask_path']) as opened:raw=np.asarray(opened.convert('L'))
            rows.append(inspect_support(raw,record['score'],record['record_id'],scope,binding,
                translation=tuple(case['crop_box_xyxy'][:2]),matrix=matrix))
        cases.append({'side':case['id'],'records':rows,'manifest_sha256':origin['manifest_sha256']})
    output=ROOT/'artifacts/mendeley_bundle_support_audit_20261005'
    if output.exists():raise FileExistsError('preserve audit')
    verify(protocol['pins'])
    if source_pins()!=protocol['mainline_pins']:raise ValueError('mainline drift during audit')
    output.mkdir(exist_ok=False)
    save(output/'report.json',{'status':'complete','cases':cases,
        'posthoc_raw_pixel_diagnostic':True,'unchanged_final_single_wire_gates':True,
        'reference_review_confirmed':False,'new_confirmed_connections':0,'deployed':False,
        'decision':'insufficient_evidence','GT_read':False,
        'source_pins':{str(p):sha256(p) for p in [Path(__file__),Path(__file__).with_name('bundle_support.py'),
            Path(__file__).with_name('core.py'),scope_path,reg_path,evidence/'protocol.json']}})
    print(json.dumps(cases,ensure_ascii=False))


if __name__=='__main__':main()
