"""Fresh5 spatial source controls, then30 development originals; no SAM/head fit."""
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
ROOT=Path(__file__).resolve().parents[2]
def main():
    out=ROOT/'artifacts/mendeley_affine_anchor_controls_20261006';out.mkdir(exist_ok=False)
    sp=ROOT/'artifacts/mendeley_reference_confirmed_20261006/reference_scope_confirmed.json'
    cp=ROOT/'artifacts/mendeley_harness_source_controls_20261006/preparation_report.json'
    dp=ROOT/'artifacts/mendeley_semantic_visible_fresh_controls_20261006/report.json'
    scope=json.loads(sp.read_text(encoding='utf-8'));reference=np.asarray(Image.open(scope['reference_image_path']).convert('RGB'))
    controls=[dict(id=c['id'],dataset='source_controls',source=c['original_source']) for c in json.loads(cp.read_text(encoding='utf-8'))['cases']]
    demo=[dict(id=c['id'],dataset='demo30',source=c['source']) for c in json.loads(dp.read_text(encoding='utf-8'))['cases'] if c['dataset']=='demo30']
    before=source_pins();pins={str(p):sha256(p) for p in [Path(__file__),sp,cp,dp,
        Path(__file__).with_name('affine_anchor_holdout.py'),Path(__file__).with_name('heldout_anchor_pose.py'),
        Path(__file__).with_name('local_anchor_pose.py')]}
    pins.update({c['source']['path']:c['source']['image_sha256'] for c in controls+demo})
    save(out/'protocol.json',dict(pins=pins,mainline_pins=before,
        model_change='8dof homography ->6dof affine only for unsupported proposals',
        matching_and_heldout_thresholds_unchanged=True,preserve_all_supported_old_poses=True,
        direct_affine_all5_source_both_anchors_required_before_demo=True,
        source_static_pixels_fresh=True,no_head_training_or_SAM=True,
        geometry_only_not_identity_or_continuity=True,repeated_development_not_field_accuracy=True,deployed=False))
    sys.path.insert(0,'E:/PythonProject10/prototype')
    from assembly_auto_review_robust_v3 import automatic_homography
    rows=[];start=time.monotonic();source_passed=None
    for case in controls+demo:
        if case['dataset']=='demo30' and source_passed is None:
            source_passed=all(p['affine']['localization_proposal_supported'] for c in rows for p in c['anchors'])
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
            chosen=old if old['localization_proposal_supported'] else affine
            anchors.append(dict(id=anchor['id'],old=old,affine=affine,chosen=chosen,
                rescued=not old['localization_proposal_supported'] and affine['localization_proposal_supported']))
        rows.append(dict(**case,global_pose_fresh=global_matrix.tolist(),anchors=anchors))
        save(out/'progress.json',dict(status='running',completed=len(rows),total=35))
    assert source_pins()==before and all(sha256(p)==d for p,d in pins.items())
    report=dict(status='complete',source_controls_passed=bool(source_passed),
        all_demo_executed=len(rows)==35,cases=rows,
        newly_supported_anchor_proposals=[dict(case=c['id'],anchor=p['id']) for c in rows if c['dataset']=='demo30' for p in c['anchors'] if p['rescued']],
        old_pose_losses=0,seconds=time.monotonic()-start,
        spatial_support_not_accuracy_or_connection=True,production_unchanged=True,deployed=False)
    save(out/'report.json',report);save(out/'progress.json',dict(status='complete'))
    print(json.dumps({k:v for k,v in report.items() if k!='cases'}),flush=True)
if __name__=='__main__':main()
