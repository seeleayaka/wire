"""Whole current test01 original-image spatial proposal audit, labels never read.

This is NOT a topology or fault accuracy evaluation. Both scoped semantic anchor
identities remain unconfirmed. No SAM inference, cached embeddings or GT input.
"""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from PIL import Image

from core import image_binding,sha256
from heldout_anchor_pose import localize,POLICY
from run_audit import source_pins
from run_review import save
from visible_lead_scope import validate_scope

ROOT=Path(__file__).resolve().parents[2]
sys.dont_write_bytecode=True


def main():
    base=ROOT/'artifacts/mendeley_visible_fan_scope_20261005'
    scope_path=ROOT/'artifacts/mendeley_reference_lead_calibration_20261005/reference_scope_draft.json'
    old=json.loads((base/'protocol.json').read_text(encoding='utf-8'))
    ref_path=Path(old['original_sources']['reference']['path'])
    folder=Path(old['original_sources']['inspection']['path']).parent
    inputs=sorted(folder.glob('*.JPG'))
    if len(inputs)!=30:raise ValueError('expected frozen whole30 test01 inputs, not selectively filtered cases')
    scope=json.loads(scope_path.read_text(encoding='utf-8'));validate_scope(scope,image_binding(ref_path))
    output=ROOT/'artifacts/mendeley_heldout_pose_all30_20261005'
    if output.exists():raise FileExistsError('new original-image run only')
    files=[Path(__file__),Path(__file__).with_name('heldout_anchor_pose.py'),
        Path(__file__).with_name('local_anchor_pose.py'),Path(__file__).with_name('prepare_mendeley_scope.py'),
        Path(__file__).with_name('visible_lead_scope.py'),scope_path,ref_path,*inputs,
        Path('E:/PythonProject10/prototype/assembly_auto_review_robust_v3.py'),
        Path('E:/PythonProject10/prototype/assembly_auto_review_robust_v2.py'),
        Path('E:/PythonProject10/prototype/assembly_auto_review_dino.py')]
    pins={str(p):sha256(p) for p in files};mainline=source_pins()
    output.mkdir(exist_ok=False)
    save(output/'protocol.json',{'created_at':datetime.now(timezone.utc).isoformat(),
        'reference':{'path':str(ref_path),**image_binding(ref_path)},
        'inputs':[{'id':f'case_{i:02d}','path':str(p),**image_binding(p)} for i,p in enumerate(inputs,1)],
        'pins':pins,'mainline_pins':mainline,'policy':POLICY,
        'input_selection':'ALL30 JPG originals in existing test01, no filtering by filename, labels or known outcomes',
        'reference_review_confirmed':False,'posthoc_development_audit':True,
        'GT_read':False,'SAM_inference':False,'decision':'insufficient_evidence',
        'stop_if':'source/code drift, exception or5min between-case budget; no threshold scan'})
    sys.path.insert(0,'E:/PythonProject10/prototype')
    from assembly_auto_review_robust_v3 import automatic_homography
    with Image.open(ref_path) as opened:reference=np.asarray(opened.convert('RGB'))
    begun=time.perf_counter();results=[]
    try:
        for i,path in enumerate(inputs,1):
            if time.perf_counter()-begun>300:raise TimeoutError('5min between-case limit')
            save(output/'progress.json',{'status':'running','current':i,'total':30,'completed':len(results),
                'seconds':time.perf_counter()-begun})
            if sha256(path)!=pins[str(path)]:raise ValueError('original input drift')
            with Image.open(path) as opened:inspection=np.asarray(opened.convert('RGB'))
            started=time.perf_counter()
            _,registration=automatic_homography(reference[:,:,::-1].copy(),inspection[:,:,::-1].copy())
            anchors=[]
            if registration['alignment_quality']['reliable']:
                matrix=np.asarray(registration['source_to_reference_homography'])
                anchors=[localize(reference,inspection,a,matrix) for a in scope['anchors']]
            results.append({'id':f'case_{i:02d}','input_path':str(path),'input_sha256':pins[str(path)],
                'registration':registration,'anchors':anchors,'seconds':time.perf_counter()-started,
                'decision':'insufficient_evidence','anchor_identity_confirmed':False,
                'new_confirmed_connections':0,'mask_inference_run':False})
        if any(sha256(p)!=d for p,d in pins.items()) or source_pins()!=mainline:raise ValueError('source/code/E drift')
        report={'status':'complete','seconds':time.perf_counter()-begun,'cases':results,
            'original_inspection_images':30,'global_registration_supported':sum(
                r['registration']['alignment_quality']['reliable'] for r in results),
            'both_anchor_spatial_proposals_supported':sum(len(r['anchors'])==2 and all(
                a['localization_proposal_supported'] for a in r['anchors']) for r in results),
            'reference_review_confirmed':False,'new_confirmed_connections':0,'deployed':False,
            'GT_read':False,'field_accuracy_not_measured':True,
            'same_device_development_images_may_overlap_prior_research':True,
            'mainline_and_inputs_unchanged':True,'fresh_SAM_encoders':0,
            'protocol_sha256':sha256(output/'protocol.json')}
        save(output/'report.json',report);save(output/'progress.json',{'status':'complete','completed':30,
            'seconds':report['seconds'],'new_confirmed_connections':0})
        print(json.dumps({k:report[k] for k in ['status','seconds','original_inspection_images',
            'global_registration_supported','both_anchor_spatial_proposals_supported','new_confirmed_connections']}))
    except BaseException as error:
        save(output/'progress.json',{'status':'failed','completed':len(results),'reason':str(error),'type':type(error).__name__})
        (output/'failure.log').write_text(traceback.format_exc(),encoding='utf-8');raise


if __name__=='__main__':main()
