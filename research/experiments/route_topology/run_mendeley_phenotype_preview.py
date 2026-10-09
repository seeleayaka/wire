"""ALL30 originals -> fresh global/local geometry -> frozen socket phenotype.

Not a full SAM/topology run, not electrical continuity or independent field test.
"""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time
import traceback
import cv2
import numpy as np
from PIL import Image
from core import sha256,image_binding
from heldout_anchor_pose import localize
from run_audit import source_pins
from run_review import save
from socket_appearance import descriptor
from socket_phenotype import predict
from socket_phenotype_agreement import feature_view,agreement
from visible_lead_scope import validate_scope

ROOT=Path(__file__).resolve().parents[2]


def main():
    source=ROOT/'artifacts/mendeley_socket_phenotype_agreement_20261005'
    audit_path=ROOT/'artifacts/mendeley_socket_phenotype_agreement_audit_20261005/report.json'
    audit=json.loads(audit_path.read_text(encoding='utf-8'))
    protocol=json.loads((source/'protocol.json').read_text(encoding='utf-8'))
    if audit['status']!='PASS' or not audit['source_gate_passed']:raise ValueError('source gate/replay required before preview')
    if any(sha256(p)!=d for p,d in audit['pins'].items()) or source_pins()!=protocol['mainline_pins']:raise ValueError('source/E drift')
    old=json.loads((ROOT/'artifacts/mendeley_visible_fan_scope_20261005/protocol.json').read_text(encoding='utf-8'))
    ref_path=Path(old['original_sources']['reference']['path']);folder=Path(old['original_sources']['inspection']['path']).parent
    paths=sorted(folder.glob('*.JPG'))
    if len(paths)!=30:raise ValueError('full30 original inspection inventory required')
    scope_path=ROOT/'artifacts/mendeley_reference_lead_calibration_20261005/reference_scope_draft.json'
    scope=json.loads(scope_path.read_text(encoding='utf-8'));validate_scope(scope,image_binding(ref_path))
    if next(a for a in scope['anchors'] if a['id']=='FAN_CPU')['bbox_xyxy']!=[1600,1000,1700,1050]:raise ValueError('scope silently changed')
    raw_models=json.loads((source/'model.json').read_text(encoding='utf-8'));models={}
    for view,m in raw_models.items():
        models[view]=({k:np.asarray(m[k]) if k in ['center','scale','weights'] else m[k] for k in ['center','scale','weights','bias']},
            {int(k):v for k,v in m['calibration'].items()})
    files=[Path(__file__),scope_path,ref_path,*paths,source/'model.json',audit_path,
        *[Path(__file__).with_name(k+'.py') for k in ['heldout_anchor_pose','local_anchor_pose','socket_appearance','socket_phenotype','socket_phenotype_agreement','visible_lead_scope']],
        *[Path('E:/PythonProject10/prototype')/(k+'.py') for k in ['assembly_auto_review_robust_v3','assembly_auto_review_robust_v2','assembly_auto_review_dino']]]
    pins={str(p):sha256(p) for p in files};mainline=source_pins()
    output=ROOT/'artifacts/mendeley_socket_phenotype_fresh30_20261005';output.mkdir(exist_ok=False)
    save(output/'protocol.json',{'created_at':datetime.now(timezone.utc).isoformat(),'pins':pins,'mainline_pins':mainline,
        'input_paths':[str(p) for p in paths],'reference_path':str(ref_path),
        'all30_original_images_decoded_fresh':True,'cached_pose_or_inspection_features_reused':False,
        'fresh_SAM_inference':False,'source_fitted_models_reused':True,'reference_review_confirmed':False,
        'filename_is_inference_feature':False,'inspection_labels_or_GT_read':False,
        'posthoc_repeated_same_device_development_preview':True,'source_gate_and_independent_replay_passed':True,
        'stop_if':'source/code/E drift, runtime error or5min between-case; no test threshold/model search'})
    sys.path.insert(0,'E:/PythonProject10/prototype')
    from assembly_auto_review_robust_v3 import automatic_homography
    with Image.open(ref_path) as im:reference=np.asarray(im.convert('RGB'))
    begun=time.perf_counter();rows=[]
    try:
        for i,path in enumerate(paths,1):
            save(output/'progress.json',{'status':'running','current':i,'total':30,'completed':len(rows),'seconds':time.perf_counter()-begun})
            if time.perf_counter()-begun>300:raise TimeoutError('5min budget')
            if sha256(path)!=pins[str(path)]:raise ValueError('original photo changed')
            with Image.open(path) as im:rgb=np.asarray(im.convert('RGB'))
            _,registration=automatic_homography(reference[:,:,::-1].copy(),rgb[:,:,::-1].copy())
            row={'id':f'case_{i:02d}','input_path':str(path),'input_sha256':pins[str(path)],'registration':registration,
                'anchors':[],'decision':'insufficient_evidence','new_confirmed_connections':0,
                'phenotype':'uncertain','confirmed_disconnections':0,'electrical_continuity':'not_assessed'}
            if registration['alignment_quality']['reliable']:
                row['anchors']=[localize(reference,rgb,a,np.asarray(registration['source_to_reference_homography'])) for a in scope['anchors']]
                socket=next(a for a in row['anchors'] if a['id']=='FAN_CPU')
                if socket['localization_proposal_supported']:
                    transform=np.array([[1,0,-1600],[0,1,-1000],[0,0,1]])@np.asarray(socket['inspection_to_reference_local'])
                    valid=cv2.warpPerspective(np.ones(rgb.shape[:2],np.uint8),transform,(100,50),flags=cv2.INTER_NEAREST)
                    if not valid.all():raise ValueError('socket patch unsupported')
                    patch=cv2.warpPerspective(rgb,transform,(100,50),flags=cv2.INTER_LINEAR)
                    patch_path=output/(row['id']+'_socket.png');Image.fromarray(patch).save(patch_path)
                    f=descriptor(patch);preds={v:predict(m,c,feature_view(f,v)) for v,(m,c) in models.items()}
                    row.update(socket_evidence=agreement(preds),phenotype=agreement(preds)['phenotype'],
                        descriptor=f.tolist(),patch_path=str(patch_path),patch_sha256=sha256(patch_path))
            rows.append(row)
        if any(sha256(p)!=d for p,d in pins.items()) or source_pins()!=mainline:raise ValueError('source/code/E drift')
        counts={k:sum(r['phenotype']==k for r in rows) for k in ['mating_body_visible','socket_contacts_exposed','uncertain']}
        result={'status':'complete','cases':rows,'phenotype_candidates':counts,'original_inspection_images':30,
            'seconds':time.perf_counter()-begun,'new_confirmed_connections':0,'confirmed_disconnections':0,
            'reference_review_confirmed':False,'not_fault_or_field_accuracy':True,'inspection_GT_read':False,
            'deployed':False,'mainline_unchanged':True,'protocol_sha256':sha256(output/'protocol.json')}
        save(output/'report.json',result);save(output/'progress.json',{'status':'complete','completed':30,'phenotype_candidates':counts})
        print(json.dumps({k:result[k] for k in ['status','seconds','phenotype_candidates','new_confirmed_connections']}))
    except BaseException as error:
        save(output/'progress.json',{'status':'failed','completed':len(rows),'reason':str(error)})
        (output/'failure.log').write_text(traceback.format_exc(),encoding='utf-8');raise


if __name__=='__main__':main()
