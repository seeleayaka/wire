"""Frozen source-only fit/calibration; never reads inspection photos/features."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image
from core import sha256
from run_audit import source_pins
from run_review import save
from socket_appearance import descriptor
from socket_phenotype import POLICY, fit, probability, predict, source_gate

ROOT=Path(__file__).resolve().parents[2]


def main():
    label_path=ROOT/'artifacts/mendeley_source_socket_visual_labels_20261005/labels.json'
    labels=json.loads(label_path.read_text(encoding='utf-8'))
    rows=labels['rows']; normal=sorted([r for r in rows if r['id'].startswith('normal_source_')],key=lambda r:r['source_sha256'])
    routing=sorted([r for r in rows if r['id'].startswith('source_routing_') and r['visual_label'] is not None],key=lambda r:r['source_sha256'])
    exposed=sorted([r for r in rows if r['visual_label']==0],key=lambda r:r['source_sha256'])
    uncertain=[r for r in rows if r['visual_label'] is None]
    fitting=normal[:80]+routing[:17]+exposed[:20]
    calibration_rows=normal[80:]+routing[17:]+exposed[20:]
    if [len(normal),len(routing),len(exposed),len(uncertain)] != [120,37,40,3]:raise ValueError('frozen source inventory differs')
    if len({r['source_sha256'] for r in rows})!=200:raise ValueError('duplicate source bytes')
    files=[label_path,Path(__file__),Path(__file__).with_name('socket_phenotype.py'),Path(__file__).with_name('socket_appearance.py')]
    pins={str(p):sha256(p) for p in files}; mainline=source_pins()
    output=ROOT/'artifacts/mendeley_socket_source_phenotype_20261005'
    if output.exists():raise FileExistsError('preserve previous experiment')
    output.mkdir(exist_ok=False)
    save(output/'protocol.json',{'created_at':datetime.now(timezone.utc).isoformat(),'policy':POLICY,
        'fit_ids':[r['id'] for r in fitting],'calibration_ids':[r['id'] for r in calibration_rows],
        'uncertain_exclusions':[r['id'] for r in uncertain], 'pins':pins,'mainline_pins':mainline,
        'source_annotation_status':labels['label_status'],'source_images_decoded_fresh':True,
        'previous_source_local_poses_reused':True,'inspection_inputs_or_labels_read':False,
        'filename_is_prediction_feature':False,'human_reference_review_confirmed':False,
        'stop_if':'any source/code/E drift or source gate failure; no inspection before source gate passes'})
    begun=time.perf_counter();prepared=[];features={}
    for phase,group in [('fit',fitting),('calibration',calibration_rows)]:
        for row in group:
            if sha256(row['source_path'])!=row['source_sha256']:raise ValueError('source byte drift')
            if not row['pose']['localization_proposal_supported']:raise ValueError('source pose unsupported')
            with Image.open(row['source_path']) as im:rgb=np.asarray(im.convert('RGB'))
            transform=np.array([[1,0,-1600],[0,1,-1000],[0,0,1]])@np.asarray(row['pose']['inspection_to_reference_local'])
            valid=cv2.warpPerspective(np.ones(rgb.shape[:2],np.uint8),transform,(100,50),flags=cv2.INTER_NEAREST)
            if not valid.all():raise ValueError('out of image interpolation')
            patch=cv2.warpPerspective(rgb,transform,(100,50),flags=cv2.INTER_LINEAR)
            if not np.array_equal(patch,np.asarray(Image.open(row['patch_path']).convert('RGB'))):raise ValueError('reviewed source patch mismatch')
            f=descriptor(patch);features[row['id']]=f
            prepared.append({'id':row['id'],'phase':phase,'source_sha256':row['source_sha256'],
                'visual_label':row['visual_label'],'descriptor':f.tolist()})
    model=fit([features[r['id']] for r in fitting],[r['visual_label'] for r in fitting])
    calibration={k:[(1-probability(model,features[r['id']])) if k==1 else probability(model,features[r['id']])
        for r in calibration_rows if r['visual_label']==k] for k in [0,1]}
    evaluated=[{'id':r['id'],'visual_label':r['visual_label'],'prediction':predict(model,calibration,features[r['id']])}
        for r in calibration_rows]
    gate=source_gate(evaluated)
    if any(sha256(p)!=d for p,d in pins.items()) or source_pins()!=mainline:raise ValueError('source/code/E drift')
    save(output/'model.json',{**{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in model.items()},
        'calibration':{str(k):v for k,v in calibration.items()},'policy':POLICY})
    report={'status':'complete','source_gate':gate,'prepared':prepared,'calibration_results':evaluated,
        'fit_counts':{'visible':97,'exposed':20},'calibration_counts':{'visible':60,'exposed':20},
        'uncertain_source_excluded':3,'inspection_inputs_or_labels_read':False,
        'source_annotation_status':labels['label_status'],'field_accuracy_not_measured':True,
        'reference_review_confirmed':False,'new_confirmed_connections':0,'deployed':False,
        'mainline_unchanged':True,'seconds':time.perf_counter()-begun,
        'protocol_sha256':sha256(output/'protocol.json')}
    save(output/'report.json',report)
    print(json.dumps({k:report[k] for k in ['status','source_gate','seconds','inspection_inputs_or_labels_read']}))


if __name__=='__main__':main()
