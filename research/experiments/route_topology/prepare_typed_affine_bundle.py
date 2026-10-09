"""Fresh reference plus ALL newly supported typed spatial candidates, no GT."""
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from core import sha256,image_binding
from heldout_anchor_pose import localize
from affine_anchor_holdout import localize_affine
from scoped_affine_pose import choose_pose
from prepare_mendeley_scope import inspection_scope
from run_audit import source_pins
from run_prompt_contrast import verify,save
from semantic_visible_socket_observer import SemanticVisibleObserver
ROOT=Path(__file__).resolve().parents[2]
def main():
    rp=ROOT/'artifacts/mendeley_scoped_affine_controls_20261006/report.json'
    ap=ROOT/'artifacts/mendeley_scoped_affine_pose_audit_20261006/report.json'
    report=json.loads(rp.read_text(encoding='utf-8'));audit=json.loads(ap.read_text(encoding='utf-8'))
    assert report['source_controls_passed'] and audit['status']=='PASS' and audit['source_report_sha256']==sha256(rp)
    previous=ROOT/'artifacts/mendeley_semantic_visible_bundle_20261006/protocol.json'
    base=json.loads(previous.read_text(encoding='utf-8'));verify(base['pins'])
    sp=Path(base['confirmed_scope_path']);scope=json.loads(sp.read_text(encoding='utf-8'))
    ids={c['case'] for c in report['newly_supported']}
    selected=[dict(id='reference',path=Path(scope['reference_image_path']))]
    selected.extend(dict(id=c['id'],path=Path(c['source']['path'])) for c in report['cases'] if c['dataset']=='demo30' and c['id'] in ids)
    assert len(selected)==3
    before=source_pins();assert before==base['mainline_pins'];observer=SemanticVisibleObserver()
    pins={**base['pins'],**observer.pins,**{str(p):sha256(p) for p in [Path(__file__),rp,ap,previous,
        *[Path(__file__).with_name(n+'.py') for n in ['affine_anchor_holdout','scoped_affine_pose']],*[c['path'] for c in selected]]}}
    out=ROOT/'artifacts/mendeley_typed_affine_bundle_20261006';out.mkdir(exist_ok=False)
    sys.path.insert(0,'E:/PythonProject10/prototype')
    from assembly_auto_review_robust_v3 import automatic_homography
    reference=np.asarray(Image.open(scope['reference_image_path']).convert('RGB'));prepared=[];sam=[]
    for c in selected:
        identity=c['id'];rgb=np.asarray(Image.open(c['path']).convert('RGB'));original=dict(path=str(c['path']),**image_binding(c['path']))
        if identity=='reference':
            registration=dict(alignment_quality=dict(reliable=True),source_to_reference_homography=np.eye(3).tolist())
            poses=[dict(id=a['id'],localization_proposal_supported=True,gates=dict(reviewed_reference=True),inspection_to_reference_local=np.eye(3).tolist()) for a in scope['anchors']]
        else:
            _,registration=automatic_homography(reference[:,:,::-1].copy(),rgb[:,:,::-1].copy());assert registration['alignment_quality']['reliable']
            matrix=np.asarray(registration['source_to_reference_homography']);poses=[]
            for a in scope['anchors']:
                old=localize(reference,rgb,a,matrix);affine=localize_affine(reference,rgb,a,matrix)
                poses.append(choose_pose(a,old,affine))
        row=dict(id=identity,original_source=original,registration=registration,anchors=poses,
            phenotype='uncertain',sam_inference_requested=False,crop_context=None)
        cpu=next(p for p in poses if p['id']=='FAN_CPU')
        if cpu['localization_proposal_supported']:
            result,patch=observer.infer_original(rgb,cpu['inspection_to_reference_local'])
            row.update(socket_evidence=result,phenotype=result['phenotype']);Image.fromarray(patch).save(out/(identity+'_socket.png'))
        ready=all(p['localization_proposal_supported'] and all(p['gates'].values()) for p in poses)
        if ready and row['phenotype']!='uncertain':
            matrix=np.asarray(registration['source_to_reference_homography'])
            box=[1380,870,1780,1270] if identity=='reference' else inspection_scope([1380,870,1780,1270],matrix,original['image_size'])
            anatomy=[1420,900,1730,1230] if identity=='reference' else inspection_scope([1420,900,1730,1230],matrix,original['image_size'])
            crop=Image.fromarray(rgb).crop(box);path=out/(identity+'_original_crop.png');crop.save(path)
            l,t,r,b=np.asarray(anatomy)-np.asarray(box[:2]*2);assert 0<=l<r<=crop.width and 0<=t<b<=crop.height
            context=dict(id=identity,source=dict(path=str(path),**image_binding(path)),original_source=original,crop_box_xyxy=box,
                positive_box_source_xyxy=anatomy,positive_box_cxcywh_normalized=[float((l+r)/(2*crop.width)),float((t+b)/(2*crop.height)),float((r-l)/crop.width),float((b-t)/crop.height)])
            row.update(crop_context=context,sam_inference_requested=True);sam.append(context);pins[str(path)]=sha256(path)
        prepared.append(row)
    assert prepared[0]['sam_inference_requested'];verify(pins);assert source_pins()==before
    save(out/'preparation_report.json',dict(status='complete',cases=prepared,all_originals_poses_features_fresh=True,
        previous_source_report_used_only_for_case_selection=True,all_spatial_gains_retained=True))
    pins[str(out/'preparation_report.json')]=sha256(out/'preparation_report.json')
    save(out/'protocol.json',dict(cases=sam,pins=pins,mainline_pins=before,sam_source=base['sam_source'],checkpoint=base['checkpoint'],
        confirmed_scope_path=str(sp),recipes=base['recipes'],retrieval_threshold=.5,whole_mask_geometry_threshold=.75,
        no_prior_image_poses_features_or_SAM_reused=True,no_bridge_pixels=True,posthoc_development_not_field_accuracy=True,deployed=False))
    print(json.dumps(dict(SAM_cases=[c['id'] for c in sam],phenotypes={c['id']:c['phenotype'] for c in prepared})))
if __name__=='__main__':main()
