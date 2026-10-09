"""Fresh five-source controls, then30 development images; no SAM/topology verdict."""
import json
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image
from core import sha256
from run_audit import source_pins
from heldout_anchor_pose import localize
from semantic_visible_socket_observer import SemanticVisibleObserver
ROOT=Path(__file__).resolve().parents[2]

def main():
    out=ROOT/'artifacts/mendeley_semantic_visible_fresh_controls_20261006';out.mkdir(exist_ok=False)
    sp=ROOT/'artifacts/mendeley_reference_confirmed_20261006/reference_scope_confirmed.json'
    cp=ROOT/'artifacts/mendeley_harness_source_controls_20261006/preparation_report.json'
    dp=ROOT/'artifacts/mendeley_compound_fresh30_20261006/report.json'
    scope=json.loads(sp.read_text(encoding='utf-8'))
    cases=[dict(id=c['id'],dataset='source_controls',source=c['original_source'],
        expected=1 if c['source_control_annotation'] is None else c['source_control_annotation']['visual_label'])
        for c in json.loads(cp.read_text(encoding='utf-8'))['cases']]
    cases.extend(dict(id=c['id'],dataset='demo30',source=c['source'],expected=None)
        for c in json.loads(dp.read_text(encoding='utf-8'))['cases'] if c['id']!='reference')
    before=source_pins();observer=SemanticVisibleObserver()
    pins={**observer.pins,**{str(p):sha256(p) for p in [sp,cp,dp,Path(__file__),
         Path(__file__).with_name('semantic_visible_socket_observer.py'),
         Path(__file__).with_name('translation_socket_observer.py'),
         Path(__file__).with_name('compound_socket_observer.py')]}}
    pins.update({c['source']['path']:c['source']['image_sha256'] for c in cases})
    (out/'protocol.json').write_text(json.dumps(dict(pins=pins,production_pins=before,
        no_prior_images_poses_features_or_masks_reused=True,
        prior_report_used_only_for_original_case_list=True,
        controls_before_demo=True,no_test_GT_read=True,
        frozen_thresholds_weights=True,no_sam=True,not_topology_acceptance=True,deployed=False),indent=2),encoding='utf-8')
    sys.path.insert(0,'E:/PythonProject10/prototype')
    from assembly_auto_review_robust_v3 import automatic_homography
    reference=np.asarray(Image.open(scope['reference_image_path']).convert('RGB'))
    anchor=next(a for a in scope['anchors'] if a['id']=='FAN_CPU');rows=[];start=time.monotonic()
    for case in cases:
        if case['dataset']=='demo30' and not all(c['prediction']['visual_label_candidate']==c['expected'] for c in rows if c['dataset']=='source_controls'):raise ValueError('five-source prerequisite failed')
        rgb=np.asarray(Image.open(case['source']['path']).convert('RGB'))
        if case['id']=='reference':pose=dict(inspection_to_reference_local=np.eye(3).tolist(),localization_proposal_supported=True)
        else:
            _,registration=automatic_homography(reference[:,:,::-1].copy(),rgb[:,:,::-1].copy())
            assert registration['alignment_quality']['reliable']
            pose=localize(reference,rgb,anchor,np.asarray(registration['source_to_reference_homography']))
        assert pose['localization_proposal_supported']
        result,patch=observer.infer_original(rgb,pose['inspection_to_reference_local'])
        Image.fromarray(patch).save(out/(case['dataset']+'_'+case['id']+'_socket.png'))
        rows.append(dict(**case,pose=pose,prediction=result))
        (out/'progress.json').write_text(json.dumps(dict(status='running',completed=len(rows),total=len(cases))),encoding='utf-8')
    assert source_pins()==before and all(sha256(p)==d for p,d in pins.items())
    controls=[c for c in rows if c['dataset']=='source_controls'];demo=[c for c in rows if c['dataset']=='demo30']
    report=dict(status='complete',cases=rows,
        source_controls_passed=all(c['prediction']['visual_label_candidate']==c['expected'] for c in controls),
        demo_counts={str(k):sum(c['prediction']['visual_label_candidate']==k for c in demo) for k in [0,1,None]},
        new_visible_candidates=[c['id'] for c in demo if c['prediction']['semantic_visible_rescue']],
        prior_supported_losses=[c['id'] for c in demo if c['prediction']['prior_compound_candidate'] is not None and c['prediction']['prior_compound_candidate']!=c['prediction']['visual_label_candidate']],
        source_and_repeated_demo_not_field_accuracy=True,production_unchanged=True,
        seconds=time.monotonic()-start,topology_acceptance=False,deployed=False)
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (out/'progress.json').write_text(json.dumps(dict(status='complete')),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='cases'}),flush=True)
if __name__=='__main__':main()
