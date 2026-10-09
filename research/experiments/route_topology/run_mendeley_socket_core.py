"""Fresh original RGB pixels + explicitly reused verified poses; normal-fit-only mask.

This is an appearance experiment, not another from-scratch registration/SAM
acceptance run. Original source bytes are decoded again; saved image patches,
SAM masks/embeddings and inspection labels are not model inputs.
"""
import json
from pathlib import Path
import time
import traceback

import cv2
import numpy as np
from PIL import Image

from core import sha256
from run_audit import source_pins
from run_review import save
from socket_appearance import fit_normal,score,novelty
from socket_core_appearance import learn_core,descriptor,POLICY

ROOT=Path(__file__).resolve().parents[2]


def main():
    prior=ROOT/'artifacts/mendeley_socket_normal_appearance_20261005'
    protocol=json.loads((prior/'protocol.json').read_text(encoding='utf-8'))
    old=json.loads((prior/'report.json').read_text(encoding='utf-8'))
    audit_path=ROOT/'artifacts/mendeley_socket_appearance_independent_audit_20261005/report.json'
    audit=json.loads(audit_path.read_text(encoding='utf-8'))
    if old['status']!='complete' or audit['status']!='PASS':raise ValueError('completed independently checked original prep required')
    for p,d in protocol['pins'].items():
        if sha256(p)!=d:raise ValueError('prior source/code drift')
    if source_pins()!=protocol['mainline_pins']:raise ValueError('E mainline drift')
    output=ROOT/'artifacts/mendeley_socket_source_core_20261005'
    if output.exists():raise FileExistsError('preserve earlier experiment')
    files=[Path(__file__),Path(__file__).with_name('socket_core_appearance.py'),
        Path(__file__).with_name('socket_appearance.py'),prior/'protocol.json',prior/'report.json',audit_path]
    pins={**protocol['pins'],**{str(p):sha256(p) for p in files}}
    output.mkdir(exist_ok=False)
    save(output/'protocol.json',{'policy':POLICY,'pins':pins,'mainline_pins':source_pins(),
        'fit_paths':protocol['fit_paths'],'calibration_paths':protocol['calibration_paths'],
        'inspection_paths':protocol['inspection_paths'],'scope':'appearance weighting only; not cable mask geometry',
        'reuses_verified_pose_transforms':True,'source_RGB_decoded_fresh':True,
        'saved_prior_image_patches_used_as_input':False,'inspection_labels_or_boxes_read':False,
        'posthoc_development_not_acceptance':True,'reference_review_confirmed':False,
        'final_acceptance_thresholds_changed':False,'stop_if':'source-derived core invalid or source/code/E drift; no retries/sweep'})
    begun=time.perf_counter();patches={'fit':[],'calibration':[],'inspection':[]};rows={'fit':[],'calibration':[],'inspection':[]}
    try:
        for row in old['prepared']:
            if not row['appearance_prepared'] or not row['pose']['localization_proposal_supported']:
                raise ValueError('missing qualified appearance prep')
            if sha256(row['path'])!=row['input_sha256']:raise ValueError('original source drift')
            with Image.open(row['path']) as opened:original=np.asarray(opened.convert('RGB'))
            matrix=np.array([[1,0,-1600],[0,1,-1000],[0,0,1]])@np.array(row['pose']['inspection_to_reference_local'])
            patch=cv2.warpPerspective(original,matrix,(100,50),flags=cv2.INTER_LINEAR)
            phase=row['phase'];patches[phase].append(patch);rows[phase].append(row)
        core,core_evidence=learn_core(patches['fit']);Image.fromarray(core.astype('uint8')*255).save(output/'source_normal_core.png')
        features={p:[descriptor(x,core) for x in group] for p,group in patches.items()}
        model=fit_normal(features['fit']);calibration=[score(model,f) for f in features['calibration']]
        cases=[]
        for row,patch,f in zip(rows['inspection'],patches['inspection'],features['inspection']):
            evidence=novelty(model,calibration,f);Image.fromarray(patch).save(output/(row['id']+'_socket.png'))
            cases.append({'id':row['id'],'path':row['path'],'source_sha256':row['input_sha256'],
                'appearance_evidence':evidence,'old_appearance_evidence':row['appearance_evidence']})
        if any(sha256(p)!=d for p,d in pins.items()) or source_pins()!=protocol['mainline_pins']:raise ValueError('input/code/E drift')
        result={'status':'complete','seconds':time.perf_counter()-begun,'core_evidence':core_evidence,
            'fit_count':80,'calibration_count':40,'inspection_count':30,'cases':cases,
            'appearance_candidates':sum(c['appearance_evidence']['appearance_difference_candidate'] for c in cases),
            'old_appearance_candidates':18,'new_confirmed_connections':0,'confirmed_disconnections':0,
            'deployed':False,'decision':'insufficient_evidence','reference_review_confirmed':False,
            'not_independent_validation_or_field_accuracy':True,'fresh_registrations':0,'fresh_SAM_encoders':0,
            'protocol_sha256':sha256(output/'protocol.json')}
        save(output/'report.json',result)
        save(output/'model.json',{'center':model['center'].tolist(),'precision':model['precision'].tolist(),
            'calibration_scores':calibration,'policy':POLICY,'core_sha256':sha256(output/'source_normal_core.png')})
        print(json.dumps({k:result[k] for k in ['status','seconds','core_evidence','appearance_candidates','old_appearance_candidates',
            'new_confirmed_connections']},ensure_ascii=False))
    except BaseException as error:
        save(output/'progress.json',{'status':'failed','reason':str(error),'type':type(error).__name__})
        (output/'failure.log').write_text(traceback.format_exc(),encoding='utf-8');raise


if __name__=='__main__':main()
