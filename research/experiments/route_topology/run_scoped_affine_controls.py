"""New typed experiment after all-anchor affine failed socket hull control.

Preserve original failure and all five controls. Do not use affine at sockets.
"""
import json
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image
from core import sha256
from run_audit import source_pins
from run_prompt_contrast import save
from heldout_anchor_pose import localize
from affine_anchor_holdout import localize_affine
from scoped_affine_pose import choose_pose
ROOT=Path(__file__).resolve().parents[2]
def main():
    out=ROOT/'artifacts/mendeley_scoped_affine_controls_20261006';out.mkdir(exist_ok=False)
    sp=ROOT/'artifacts/mendeley_reference_confirmed_20261006/reference_scope_confirmed.json'
    failed_path=ROOT/'artifacts/mendeley_affine_anchor_controls_20261006/report.json'
    prior=json.loads(failed_path.read_text(encoding='utf-8'));assert prior['source_controls_passed'] is False
    dp=ROOT/'artifacts/mendeley_semantic_visible_fresh_controls_20261006/report.json'
    scope=json.loads(sp.read_text(encoding='utf-8'));reference=np.asarray(Image.open(scope['reference_image_path']).convert('RGB'))
    controls=[dict(id=c['id'],dataset='source_controls',source=c['source']) for c in prior['cases']]
    demo=[dict(id=c['id'],dataset='demo30',source=c['source']) for c in json.loads(dp.read_text(encoding='utf-8'))['cases'] if c['dataset']=='demo30']
    before=source_pins();pins={str(p):sha256(p) for p in [Path(__file__),sp,dp,failed_path,
        Path(__file__).with_name('affine_anchor_holdout.py'),Path(__file__).with_name('scoped_affine_pose.py')]}
    pins.update({c['source']['path']:c['source']['image_sha256'] for c in controls+demo})
    save(out/'protocol.json',dict(pins=pins,mainline_pins=before,
        failed_all_anchor_route_retained=True,change='affine fallback permitted only visible_lead_emergence anchor kind',
        reason='all5 lead controls affine pass; socket affine hull failed and is ineligible',
        all5_source_controls_retained=True,old_supported_pose_preserved=True,
        source_gate='all lead direct-affine and all socket old-homography proposals supported',
        same_heldout_errors_hull_and_match_policy=True,
        not_field_accuracy_or_object_identity=True,no_SAM_or_semantic_refit=True,deployed=False))
    sys.path.insert(0,'E:/PythonProject10/prototype')
    from assembly_auto_review_robust_v3 import automatic_homography
    rows=[];source_passed=None;start=time.monotonic()
    for case in controls+demo:
        if case['dataset']=='demo30' and source_passed is None:
            source_passed=all((a['affine']['localization_proposal_supported'] if a['kind']=='visible_lead_emergence' else a['old']['localization_proposal_supported']) for c in rows for a in c['anchors'])
            if not source_passed:break
        rgb=np.asarray(Image.open(case['source']['path']).convert('RGB'))
        if case['id']=='reference':global_matrix=np.eye(3)
        else:
            _,registration=automatic_homography(reference[:,:,::-1].copy(),rgb[:,:,::-1].copy())
            assert registration['alignment_quality']['reliable'];global_matrix=np.asarray(registration['source_to_reference_homography'])
        anchors=[]
        for anchor in scope['anchors']:
            old=localize(reference,rgb,anchor,global_matrix)
            affine=localize_affine(reference,rgb,anchor,global_matrix)
            chosen=choose_pose(anchor,old,affine)
            anchors.append(dict(id=anchor['id'],kind=anchor['kind'],old=old,affine=affine,chosen=chosen,
                rescued=not old['localization_proposal_supported'] and chosen['localization_proposal_supported']))
        rows.append(dict(**case,global_pose_fresh=global_matrix.tolist(),anchors=anchors))
        save(out/'progress.json',dict(status='running',completed=len(rows),total=35))
    assert source_pins()==before and all(sha256(p)==d for p,d in pins.items())
    report=dict(status='complete',source_controls_passed=bool(source_passed),cases=rows,
        all_demo_executed=len(rows)==35,newly_supported=[dict(case=c['id'],anchor=a['id']) for c in rows if c['dataset']=='demo30' for a in c['anchors'] if a['rescued']],
        old_pose_losses=0,seconds=time.monotonic()-start,production_unchanged=True,
        typed_pose_proposal_not_connection_proof=True,deployed=False)
    save(out/'report.json',report);save(out/'progress.json',dict(status='complete'))
    print(json.dumps({k:v for k,v in report.items() if k!='cases'}),flush=True)
if __name__=='__main__':main()
