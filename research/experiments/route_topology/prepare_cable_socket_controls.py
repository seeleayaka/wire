"""Fixed source-first cable-mask/socket-conflict falsification, no demo tuning."""
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from core import image_binding, sha256
from heldout_anchor_pose import localize
from affine_anchor_holdout import localize_affine
from scoped_affine_pose import choose_pose
from prepare_mendeley_scope import inspection_scope
from semantic_visible_socket_observer import SemanticVisibleObserver
from run_audit import source_pins
from run_prompt_contrast import save, verify

ROOT=Path(__file__).resolve().parents[2]

def main():
    prior_path=ROOT/'artifacts/mendeley_scoped_affine_controls_20261006/report.json'
    prior=json.loads(prior_path.read_text(encoding='utf-8'))
    assert prior['source_controls_passed']
    controls=[c for c in prior['cases'] if c['dataset']=='source_controls']
    assert len(controls)==5
    previous=ROOT/'artifacts/mendeley_typed_affine_bundle_20261006/protocol.json'
    base=json.loads(previous.read_text(encoding='utf-8')); verify(base['pins'])
    sp=Path(base['confirmed_scope_path']); scope=json.loads(sp.read_text(encoding='utf-8'))
    before=source_pins(); assert before==base['mainline_pins']
    observer=SemanticVisibleObserver()
    files=[Path(__file__),prior_path,previous,sp,*[Path(__file__).with_name(n+'.py') for n in
        ['heldout_anchor_pose','affine_anchor_holdout','scoped_affine_pose','run_mendeley_geometry','visible_bundle_relation']]]
    pins={**base['pins'],**observer.pins,**{str(p):sha256(p) for p in files}}
    out=ROOT/'artifacts/mendeley_cable_socket_controls_20261006'; out.mkdir(exist_ok=False)
    save(out/'experiment_contract.json',dict(
        hypothesis='high-score cable-mask socket-box overlap discriminates visible body from exposed socket',
        source_inventory='all five pre-existing controls; none omitted; no demo images',
        expected_visible_ids=[c['id'] for c in controls if c['id']=='reference' or 'visible' in c['id']],
        expected_exposed_ids=[c['id'] for c in controls if 'exposed' in c['id']],
        falsification='any exposed control with high-score socket-box overlap refutes specificity; absence is not disconnection proof',
        original_pose_gate_unchanged=True,fragment_pixels_retained=True,
        no_parameter_sweep=True,no_decision_override=True,no_demo_extension=True,
        qualitative_source_controls_not_human_GT=True,no_deployment=True))
    pins[str(out/'experiment_contract.json')]=sha256(out/'experiment_contract.json')
    sys.path.insert(0,'E:/PythonProject10/prototype')
    from assembly_auto_review_robust_v3 import automatic_homography
    reference=np.asarray(Image.open(scope['reference_image_path']).convert('RGB'))
    prepared=[]; sam=[]
    for case in controls:
        identity=case['id']; path=Path(case['source']['path'])
        assert sha256(path)==case['source']['image_sha256']; pins[str(path)]=sha256(path)
        rgb=np.asarray(Image.open(path).convert('RGB')); original=dict(path=str(path),**image_binding(path))
        if identity=='reference':
            registration=dict(alignment_quality=dict(reliable=True),source_to_reference_homography=np.eye(3).tolist())
            poses=[dict(id=a['id'],localization_proposal_supported=True,gates=dict(reviewed_reference=True),inspection_to_reference_local=np.eye(3).tolist()) for a in scope['anchors']]
        else:
            _,registration=automatic_homography(reference[:,:,::-1].copy(),rgb[:,:,::-1].copy())
            assert registration['alignment_quality']['reliable']
            matrix=np.asarray(registration['source_to_reference_homography'])
            poses=[choose_pose(a,localize(reference,rgb,a,matrix),localize_affine(reference,rgb,a,matrix)) for a in scope['anchors']]
        assert all(p['localization_proposal_supported'] and all(p['gates'].values()) for p in poses)
        cpu=next(p for p in poses if p['id']=='FAN_CPU')
        result,patch=observer.infer_original(rgb,cpu['inspection_to_reference_local'])
        expected='socket_contacts_exposed' if 'exposed' in identity else 'mating_body_visible'
        assert result['phenotype']==expected, 'source appearance control failed; no SAM expansion'
        Image.fromarray(patch).save(out/(identity+'_socket.png'))
        matrix=np.asarray(registration['source_to_reference_homography'])
        box=[1380,870,1780,1270] if identity=='reference' else inspection_scope([1380,870,1780,1270],matrix,original['image_size'])
        anatomy=[1420,900,1730,1230] if identity=='reference' else inspection_scope([1420,900,1730,1230],matrix,original['image_size'])
        crop=Image.fromarray(rgb).crop(box); crop_path=out/(identity+'_original_crop.png'); crop.save(crop_path)
        l,t,r,b=np.asarray(anatomy)-np.asarray(box[:2]*2)
        assert 0<=l<r<=crop.width and 0<=t<b<=crop.height
        context=dict(id=identity,source=dict(path=str(crop_path),**image_binding(crop_path)),original_source=original,crop_box_xyxy=box,
            positive_box_source_xyxy=anatomy,positive_box_cxcywh_normalized=[float((l+r)/(2*crop.width)),float((t+b)/(2*crop.height)),float((r-l)/crop.width),float((b-t)/crop.height)])
        pins[str(crop_path)]=sha256(crop_path); sam.append(context)
        prepared.append(dict(id=identity,original_source=original,registration=registration,anchors=poses,
            socket_evidence=result,phenotype=result['phenotype'],sam_inference_requested=True,crop_context=context))
    verify(pins); assert source_pins()==before
    save(out/'preparation_report.json',dict(status='complete',cases=prepared,originals_poses_features_fresh=True,all5_controls_retained=True))
    pins[str(out/'preparation_report.json')]=sha256(out/'preparation_report.json')
    save(out/'protocol.json',dict(cases=sam,pins=pins,mainline_pins=before,sam_source=base['sam_source'],checkpoint=base['checkpoint'],
        confirmed_scope_path=str(sp),recipes=base['recipes'],retrieval_threshold=.5,whole_mask_geometry_threshold=.75,
        no_prior_SAM_or_inspection_poses_reused=True,no_bridge_pixels=True,no_demo_extension=True,deployed=False))
    print(json.dumps(dict(preparation='complete',SAM_cases=[c['id'] for c in sam])))

if __name__=='__main__':main()
