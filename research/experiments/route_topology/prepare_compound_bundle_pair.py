"""Confirmed reference -> fresh original RGB/poses/socket -> frozen SAM protocol.

Three qualitative demo photos selected after prior review, NOT a new blind test.
No inspection embeddings, images, patches or pose matrices are reused.
"""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import numpy as np
import cv2
from PIL import Image
from core import sha256,image_binding
from heldout_anchor_pose import localize
from prepare_mendeley_scope import inspection_scope
from run_audit import source_pins
from run_prompt_contrast import verify
from run_review import save
from socket_appearance import descriptor
from socket_phenotype import predict
from socket_phenotype_agreement import feature_view,agreement
from visible_lead_scope import validate_scope
from compound_socket_observer import Observer

ROOT=Path(__file__).resolve().parents[2]


def main():
    approval_folder=ROOT/'artifacts/mendeley_reference_confirmed_20261006'
    scope_path=approval_folder/'reference_scope_confirmed.json'
    scope=json.loads(scope_path.read_text(encoding='utf-8'))
    approval=json.loads((approval_folder/'approval_record.json').read_text(encoding='utf-8'))
    ref_path=Path(scope['reference_image_path']);binding=image_binding(ref_path)
    if not validate_scope(scope,binding) or sha256(scope_path)!=approval['confirmed_scope_sha256']:raise ValueError('confirmed image-bound scope required')
    if source_pins()!=approval['mainline_pins']:raise ValueError('E changed since reference confirmation')
    old_path=ROOT/'artifacts/mendeley_reference_geometry_prompt_20261005/protocol.json'
    old=json.loads(old_path.read_text(encoding='utf-8'));verify(old['pins'])
    model_path=ROOT/'artifacts/mendeley_socket_phenotype_agreement_20261005/model.json'
    audit_path=ROOT/'artifacts/mendeley_socket_phenotype_agreement_audit_20261005/report.json'
    audit=json.loads(audit_path.read_text(encoding='utf-8'))
    if audit['status']!='PASS' or not audit['source_gate_passed'] or any(sha256(p)!=d for p,d in audit['pins'].items()):raise ValueError('source phenotype replay/gate required')
    raw_models=json.loads(model_path.read_text(encoding='utf-8'))
    models={v:({k:np.asarray(m[k]) if k in ['center','scale','weights'] else m[k] for k in ['center','scale','weights','bias']},
        {int(k):s for k,s in m['calibration'].items()}) for v,m in raw_models.items()}
    base=json.loads((ROOT/'artifacts/mendeley_visible_fan_scope_20261005/protocol.json').read_text(encoding='utf-8'))
    paths=sorted(Path(base['original_sources']['inspection']['path']).parent.glob('*.JPG'))
    if len(paths)!=30:raise ValueError('demo source inventory changed')
    selected=[('reference',ref_path),('case_08',paths[7])]
    composition_audit_path=ROOT/'artifacts/mendeley_compound_observer_audit_20261006/report.json'
    composition_audit=json.loads(composition_audit_path.read_text(encoding='utf-8'))
    if composition_audit['status']!='PASS':raise ValueError('compound replay required')
    observer=Observer()
    files=[Path(__file__),scope_path,approval_folder/'approval_record.json',model_path,audit_path,old_path,
        *[Path(__file__).with_name(n+'.py') for n in ['run_mendeley_geometry','run_prompt_contrast','heldout_anchor_pose',
            'local_anchor_pose','socket_appearance','socket_phenotype','socket_phenotype_agreement',
            'visible_bundle_relation','visible_lead_scope','prepare_mendeley_scope','compound_socket_observer','component_cues','run_positive_contact_gate']],
        *[p for _,p in selected],*[Path('E:/PythonProject10/prototype')/(n+'.py') for n in
            ['assembly_auto_review_robust_v3','assembly_auto_review_robust_v2','assembly_auto_review_dino']]]
    pins={**old['pins'],**observer.pins,**{str(p):sha256(p) for p in files},str(composition_audit_path):sha256(composition_audit_path)};mainline=source_pins()
    output=ROOT/'artifacts/mendeley_compound_bundle_pair_20261006';output.mkdir(exist_ok=False)
    sys.path.insert(0,'E:/PythonProject10/prototype')
    from assembly_auto_review_robust_v3 import automatic_homography
    with Image.open(ref_path) as im:reference=np.asarray(im.convert('RGB'))
    prepared=[];sam_cases=[]
    for identity,path in selected:
        with Image.open(path) as im:rgb=np.asarray(im.convert('RGB'))
        original={'path':str(path),**image_binding(path)}
        if identity=='reference':
            registration={'alignment_quality':{'reliable':True},'source_to_reference_homography':np.eye(3).tolist(),'same_reviewed_reference_image':True}
            poses=[{'id':a['id'],'localization_proposal_supported':True,'gates':{'same_reviewed_reference_pixels':True},
                'inspection_to_reference_local':np.eye(3).tolist(),'reference_annotation_not_inspection_GT':True} for a in scope['anchors']]
        else:
            _,registration=automatic_homography(reference[:,:,::-1].copy(),rgb[:,:,::-1].copy())
            poses=[localize(reference,rgb,a,np.array(registration['source_to_reference_homography'])) for a in scope['anchors']] if registration['alignment_quality']['reliable'] else []
        socket=next((a for a in poses if a['id']=='FAN_CPU'),None)
        row={'id':identity,'original_source':original,'registration':registration,'anchors':poses,'phenotype':'uncertain',
            'sam_inference_requested':False,'global_and_local_inspection_geometry_fresh':identity!='reference'}
        if socket is not None and socket['localization_proposal_supported']:
            transform=np.array([[1,0,-1600],[0,1,-1000],[0,0,1]])@np.array(socket['inspection_to_reference_local'])
            valid=cv2.warpPerspective(np.ones(rgb.shape[:2],np.uint8),transform,(100,50),flags=cv2.INTER_NEAREST)
            if not valid.all():raise ValueError('socket appearance outside original pixels')
            patch=cv2.warpPerspective(rgb,transform,(100,50),flags=cv2.INTER_LINEAR)
            patch_path=output/(identity+'_socket.png');Image.fromarray(patch).save(patch_path)
            result=observer.infer(patch)
            row.update(socket_evidence=result,phenotype=result['phenotype'],
                socket_patch_path=str(patch_path),socket_patch_sha256=sha256(patch_path))
        if registration['alignment_quality']['reliable']:
            h=np.array(registration['source_to_reference_homography'])
            crop_box=[1380,870,1780,1270] if identity=='reference' else inspection_scope([1380,870,1780,1270],h,original['image_size'])
            anatomy=[1420,900,1730,1230] if identity=='reference' else inspection_scope([1420,900,1730,1230],h,original['image_size'])
            crop=Image.fromarray(rgb).crop(crop_box);crop_path=output/(identity+'_original_crop.png');crop.save(crop_path)
            local=np.array(anatomy)-np.array(crop_box[:2]*2)
            if not (0<=local[0]<local[2]<=crop.width and 0<=local[1]<local[3]<=crop.height):raise ValueError('reference anatomy outside exact crop, never clip')
            box=[float((local[0]+local[2])/(2*crop.width)),float((local[1]+local[3])/(2*crop.height)),
                float((local[2]-local[0])/crop.width),float((local[3]-local[1])/crop.height)]
            case={'id':identity,'source':{'path':str(crop_path),**image_binding(crop_path)},'original_source':original,
                'crop_box_xyxy':crop_box,'positive_box_cxcywh_normalized':box,'positive_box_source_xyxy':anatomy}
            row['crop_context']=case
            if row['phenotype']!='uncertain':sam_cases.append(case);row['sam_inference_requested']=True
        prepared.append(row)
    if prepared[0]['phenotype']!='mating_body_visible':raise ValueError('reference phenotype not ready; do not override model')
    if any(sha256(p)!=d for p,d in pins.items()) or source_pins()!=mainline:raise ValueError('code/input/E changed during fresh preparation')
    pins.update({str(output/(r['id']+'_original_crop.png')):sha256(output/(r['id']+'_original_crop.png')) for r in prepared if 'crop_context' in r})
    save(output/'preparation_report.json',{'status':'complete','cases':prepared,'all_original_images_decoded_fresh':True,
        'inspection_pose_or_features_reused':False,'reference_review_confirmed':True,
        'source_fitted_models_reused':True,'inspection_labels_or_GT_read':False,
        'sample_selected_posthoc_for_demo_not_independent_accuracy':True})
    pins[str(output/'preparation_report.json')]=sha256(output/'preparation_report.json')
    save(output/'protocol.json',{'created_at_utc':datetime.now(timezone.utc).isoformat(),'client_date_hk':'2026-10-06',
        'cases':sam_cases,'pins':pins,'mainline_pins':mainline,'sam_source':old['sam_source'],'checkpoint':old['checkpoint'],
        'recipes':['cable','cable_plus_reference_anatomy_box'],'retrieval_threshold':.5,'whole_mask_geometry_threshold':.75,
        'reference_anatomy_xyxy':[1420,900,1730,1230],'confirmed_scope_path':str(scope_path),
        'source_models_frozen':True,'original_RGB_crops_no_pixel_warp':True,'fresh_image_encoder_required':True,
        'prior_inspection_SAM_or_embeddings_reused':False,'reference_review_confirmed':True,
        'model_observer_count':1,'new_bundle_type_not_single_wire_gate_change':True,
        'uncertain_socket_skips_SAM_and_abstains':True,'GT_read':False,
        'sample_selection':'reference plus new appearance candidate08; posthoc research not blind validation',
        'stop_if':'source/code/E drift, error or30min between-case; no prompt/threshold/model sweep'})
    print(json.dumps({'protocol':str(output/'protocol.json'),'fresh_originals':len(prepared),
        'new_SAM_cases':[c['id'] for c in sam_cases],'socket_states':{r['id']:r['phenotype'] for r in prepared}},ensure_ascii=False))


if __name__=='__main__':main()


