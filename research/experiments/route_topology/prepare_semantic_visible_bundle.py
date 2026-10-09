"""Fresh reference and ALL four new appearance candidates, scoped bundle check.

No previous inspection poses/features/masks reused. Unsupported FAN_LEAD remains
unknown and skips SAM; do not equate localized socket state with a graph edge.
"""
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from core import sha256,image_binding
from heldout_anchor_pose import localize
from prepare_mendeley_scope import inspection_scope
from run_audit import source_pins
from run_prompt_contrast import verify,save
from visible_lead_scope import validate_scope
from semantic_visible_socket_observer import SemanticVisibleObserver
ROOT=Path(__file__).resolve().parents[2]

def main():
    sp=ROOT/'artifacts/mendeley_reference_confirmed_20261006/reference_scope_confirmed.json'
    ap=sp.parent/'approval_record.json';scope=json.loads(sp.read_text(encoding='utf-8'))
    approval=json.loads(ap.read_text(encoding='utf-8'));reference_path=Path(scope['reference_image_path'])
    assert validate_scope(scope,image_binding(reference_path)) and sha256(sp)==approval['confirmed_scope_sha256']
    assert source_pins()==approval['mainline_pins']
    rp=ROOT/'artifacts/mendeley_semantic_visible_fresh_controls_20261006/report.json'
    audit_path=ROOT/'artifacts/mendeley_semantic_visible_fresh_audit_20261006/report.json'
    fresh=json.loads(rp.read_text(encoding='utf-8'));audit=json.loads(audit_path.read_text(encoding='utf-8'))
    assert fresh['source_controls_passed'] and audit['status']=='PASS' and audit['source_report_sha256']==sha256(rp)
    selected=[dict(id='reference',path=reference_path)]
    selected.extend(dict(id=c['id'],path=Path(c['source']['path'])) for c in fresh['cases'] if c['prediction']['semantic_visible_rescue'])
    assert len(selected)==5
    old_path=ROOT/'artifacts/mendeley_reference_geometry_prompt_20261005/protocol.json'
    old=json.loads(old_path.read_text(encoding='utf-8'));verify(old['pins'])
    observer=SemanticVisibleObserver();mainline=source_pins()
    pins={**old['pins'],**observer.pins,**{str(p):sha256(p) for p in [Path(__file__),sp,ap,rp,audit_path,old_path,
        *[Path(__file__).with_name(n+'.py') for n in ['run_mendeley_geometry','heldout_anchor_pose','local_anchor_pose',
           'semantic_visible_socket_observer','translation_socket_observer','compound_socket_observer','visible_bundle_relation']],
        *[c['path'] for c in selected]]}}
    out=ROOT/'artifacts/mendeley_semantic_visible_bundle_20261006';out.mkdir(exist_ok=False)
    sys.path.insert(0,'E:/PythonProject10/prototype')
    from assembly_auto_review_robust_v3 import automatic_homography
    reference=np.asarray(Image.open(reference_path).convert('RGB'));prepared=[];sam_cases=[]
    for selected_case in selected:
        identity=selected_case['id'];path=selected_case['path'];rgb=np.asarray(Image.open(path).convert('RGB'))
        original=dict(path=str(path),**image_binding(path))
        if identity=='reference':
            registration=dict(alignment_quality=dict(reliable=True),source_to_reference_homography=np.eye(3).tolist())
            poses=[dict(id=a['id'],localization_proposal_supported=True,gates=dict(reviewed_reference=True),
                inspection_to_reference_local=np.eye(3).tolist()) for a in scope['anchors']]
        else:
            _,registration=automatic_homography(reference[:,:,::-1].copy(),rgb[:,:,::-1].copy())
            poses=[localize(reference,rgb,a,np.asarray(registration['source_to_reference_homography'])) for a in scope['anchors']] if registration['alignment_quality']['reliable'] else []
        row=dict(id=identity,original_source=original,registration=registration,anchors=poses,
            phenotype='uncertain',sam_inference_requested=False,crop_context=None)
        socket=next((p for p in poses if p['id']=='FAN_CPU'),None)
        if socket and socket['localization_proposal_supported']:
            prediction,patch=observer.infer_original(rgb,socket['inspection_to_reference_local'])
            row.update(socket_evidence=prediction,phenotype=prediction['phenotype'])
            Image.fromarray(patch).save(out/(identity+'_socket.png'))
        ready=(len(poses)==len(scope['anchors']) and all(p['localization_proposal_supported'] and all(p['gates'].values()) for p in poses))
        if ready and row['phenotype']=='mating_body_visible':
            h=np.asarray(registration['source_to_reference_homography'])
            crop_box=[1380,870,1780,1270] if identity=='reference' else inspection_scope([1380,870,1780,1270],h,original['image_size'])
            anatomy=[1420,900,1730,1230] if identity=='reference' else inspection_scope([1420,900,1730,1230],h,original['image_size'])
            crop=Image.fromarray(rgb).crop(crop_box);crop_path=out/(identity+'_original_crop.png');crop.save(crop_path)
            l,t,r,b=np.asarray(anatomy)-np.asarray(crop_box[:2]*2)
            assert 0<=l<r<=crop.width and 0<=t<b<=crop.height
            case=dict(id=identity,source=dict(path=str(crop_path),**image_binding(crop_path)),original_source=original,
                crop_box_xyxy=crop_box,positive_box_source_xyxy=anatomy,
                positive_box_cxcywh_normalized=[float((l+r)/(2*crop.width)),float((t+b)/(2*crop.height)),
                    float((r-l)/crop.width),float((b-t)/crop.height)])
            pins[str(crop_path)]=sha256(crop_path);sam_cases.append(case)
            row.update(crop_context=case,sam_inference_requested=True)
        prepared.append(row)
    assert prepared[0]['sam_inference_requested'] and source_pins()==mainline
    verify(pins)
    save(out/'preparation_report.json',dict(status='complete',cases=prepared,
        all_original_pose_and_features_fresh=True,all_new_candidates_retained=True,
        unsupported_anchor_skips_SAM_not_claimed_correct=True))
    pins[str(out/'preparation_report.json')]=sha256(out/'preparation_report.json')
    save(out/'protocol.json',dict(cases=sam_cases,pins=pins,mainline_pins=mainline,
        sam_source=old['sam_source'],checkpoint=old['checkpoint'],confirmed_scope_path=str(sp),
        recipes=['cable','cable_plus_reference_anatomy_box'],retrieval_threshold=.5,
        whole_mask_geometry_threshold=.75,reference_anatomy_xyxy=[1420,900,1730,1230],
        sample_selection='ALL new visible candidates; only both locally supported anchors permit SAM',
        no_prior_image_features_poses_or_SAM_reused=True,no_mask_gap_filling=True,
        posthoc_repeated_development_not_field_accuracy=True,model_observer_count=1,deployed=False))
    print(json.dumps(dict(fresh_originals=len(prepared),SAM_cases=[c['id'] for c in sam_cases],
        unsupported=[r['id'] for r in prepared if not r['sam_inference_requested']])))
if __name__=='__main__':main()
