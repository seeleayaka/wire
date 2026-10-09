"""Fresh originals -> register -> heldout socket pose -> normal-only appearance."""
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
from socket_appearance import descriptor,fit_normal,novelty,score,POLICY

ROOT=Path(__file__).resolve().parents[2]
sys.dont_write_bytecode=True


def main():
    base=ROOT/'artifacts/mendeley_visible_fan_scope_20261005'
    scope_path=ROOT/'artifacts/mendeley_reference_lead_calibration_20261005/reference_scope_draft.json'
    old=json.loads((base/'protocol.json').read_text(encoding='utf-8'))
    scope=json.loads(scope_path.read_text(encoding='utf-8'))
    anchor=next(a for a in scope['anchors'] if a['id']=='FAN_CPU')
    if anchor['bbox_xyxy']!=[1600,1000,1700,1050]:raise ValueError('calibration scope differs; do not silently recrop')
    ref_path=Path(old['original_sources']['reference']['path'])
    normal_paths=sorted(ref_path.parent.glob('normal_*.JPG'),key=sha256)
    test_paths=sorted(Path(old['original_sources']['inspection']['path']).parent.glob('*.JPG'))
    if len(normal_paths)!=120 or len(test_paths)!=30:raise ValueError('frozen full normal120/test30 expected')
    if len({sha256(p) for p in normal_paths})!=120:raise ValueError('duplicate normal photo bytes')
    fit_paths=normal_paths[:80];calibration_paths=normal_paths[80:]
    files=[Path(__file__),Path(__file__).with_name('socket_appearance.py'),
        Path(__file__).with_name('heldout_anchor_pose.py'),Path(__file__).with_name('local_anchor_pose.py'),
        Path(__file__).with_name('prepare_mendeley_scope.py'),scope_path,ref_path,*normal_paths,*test_paths,
        Path('E:/PythonProject10/prototype/assembly_auto_review_robust_v3.py'),
        Path('E:/PythonProject10/prototype/assembly_auto_review_robust_v2.py'),
        Path('E:/PythonProject10/prototype/assembly_auto_review_dino.py')]
    pins={str(p):sha256(p) for p in files};mainline=source_pins()
    output=ROOT/'artifacts/mendeley_socket_normal_appearance_20261005'
    if output.exists():raise FileExistsError('preserve previous trial')
    output.mkdir(exist_ok=False)
    save(output/'protocol.json',{'created_at':datetime.now(timezone.utc).isoformat(),
        'policy':POLICY,'pins':pins,'mainline_pins':mainline,'reference_path':str(ref_path),
        'fit_paths':[str(p) for p in fit_paths],'calibration_paths':[str(p) for p in calibration_paths],
        'inspection_paths':[str(p) for p in test_paths],
        'uses_dataset_normal_training_metadata':True,'inspection_labels_or_boxes_read':False,
        'filename_is_inference_feature':False,'source_images_decoded_fresh':True,
        'statistical_outlier_is_not_unplugged_diagnosis':True,'reference_review_confirmed':False,
        'stop_if':'any normal prep missing, source/code/E drift, runtime error or15min between-case; no threshold sweeps',
        'posthoc_same_device_development_trial':True,'fresh_SAM_inference':False})
    sys.path.insert(0,'E:/PythonProject10/prototype')
    from assembly_auto_review_robust_v3 import automatic_homography
    with Image.open(ref_path) as opened:reference=np.asarray(opened.convert('RGB'))
    begun=time.perf_counter();prepared=[];fit_features=[];calibration_features=[];cases=[]
    try:
        for phase,paths in [('fit',fit_paths),('calibration',calibration_paths),('inspection',test_paths)]:
            for i,path in enumerate(paths,1):
                if time.perf_counter()-begun>900:raise TimeoutError('15min between-case budget')
                save(output/'progress.json',{'status':'fresh_original_preparation','phase':phase,'index':i,
                    'total':len(paths),'completed':len(prepared),'seconds':time.perf_counter()-begun})
                if sha256(path)!=pins[str(path)]:raise ValueError('source drift')
                with Image.open(path) as opened:inspection=np.asarray(opened.convert('RGB'))
                if sha256(path)==sha256(ref_path):
                    registration={'alignment_quality':{'reliable':True},'source_to_reference_homography':np.eye(3).tolist(),
                        'self_control':True}
                else:_,registration=automatic_homography(reference[:,:,::-1].copy(),inspection[:,:,::-1].copy())
                row={'phase':phase,'id':f'{phase}_{i:03d}','path':str(path),'input_sha256':pins[str(path)],
                    'registration':registration,'decision':'insufficient_evidence','appearance_prepared':False}
                if registration['alignment_quality']['reliable']:
                    pose=localize(reference,inspection,anchor,np.array(registration['source_to_reference_homography']))
                    row['pose']=pose
                    if pose['localization_proposal_supported']:
                        transform=np.array([[1,0,-1600],[0,1,-1000],[0,0,1]])@np.array(pose['inspection_to_reference_local'])
                        valid=cv2.warpPerspective(np.ones(inspection.shape[:2],np.uint8),transform,(100,50),flags=cv2.INTER_NEAREST)
                        if not np.all(valid):raise ValueError('appearance interpolation outside actual source')
                        patch=cv2.warpPerspective(inspection,transform,(100,50),flags=cv2.INTER_LINEAR)
                        Image.fromarray(patch).save(output/(row['id']+'_socket.png'))
                        f=descriptor(patch);row['descriptor']=f.tolist();row['appearance_prepared']=True
                        if phase=='fit':fit_features.append(f)
                        elif phase=='calibration':calibration_features.append(f)
                if not row['appearance_prepared'] and phase!='inspection':
                    save(output/'failed_normal_preparation.json',row)
                    raise ValueError('normal calibration incomplete; do not silently discard or lower pose gates')
                prepared.append(row)
                if phase=='inspection':cases.append(row)
        model=fit_normal(fit_features);calibration_scores=[score(model,f) for f in calibration_features]
        for row in cases:
            if row['appearance_prepared']:row['appearance_evidence']=novelty(model,calibration_scores,row['descriptor'])
        if any(sha256(p)!=d for p,d in pins.items()) or source_pins()!=mainline:raise ValueError('source/code/E drift')
        save(output/'model.json',{'center':model['center'].tolist(),'precision':model['precision'].tolist(),
            'fit_count':model['fit_count'],'calibration_scores':calibration_scores,'policy':POLICY})
        report={'status':'complete','seconds':time.perf_counter()-begun,'prepared':prepared,'cases':cases,
            'fit_count':80,'calibration_count':40,'inspection_count':30,
            'appearance_difference_candidates':sum(r.get('appearance_evidence',{}).get('appearance_difference_candidate',False) for r in cases),
            'new_confirmed_connections':0,'confirmed_disconnections':0,'deployed':False,
            'reference_review_confirmed':False,'inspection_labels_or_boxes_read':False,
            'field_accuracy_not_measured':True,'mainline_unchanged':True,
            'protocol_sha256':sha256(output/'protocol.json')}
        save(output/'report.json',report);save(output/'progress.json',{'status':'complete','seconds':report['seconds'],
            'appearance_difference_candidates':report['appearance_difference_candidates'],'new_confirmed_connections':0})
        print(json.dumps({k:report[k] for k in ['status','seconds','fit_count','calibration_count','inspection_count',
            'appearance_difference_candidates','new_confirmed_connections']}))
    except BaseException as error:
        save(output/'progress.json',{'status':'failed','completed':len(prepared),'reason':str(error),'type':type(error).__name__})
        (output/'failure.log').write_text(traceback.format_exc(),encoding='utf-8');raise


if __name__=='__main__':main()
