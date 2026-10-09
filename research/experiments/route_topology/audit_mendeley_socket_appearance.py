"""Independent pixel reconstruction and statistical replay, not fault scoring."""
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

ROOT=Path(__file__).resolve().parents[2]


def main():
    source=ROOT/'artifacts/mendeley_socket_normal_appearance_20261005'
    protocol=json.loads((source/'protocol.json').read_text(encoding='utf-8'))
    report=json.loads((source/'report.json').read_text(encoding='utf-8'))
    model_path=source/'model.json';model=json.loads(model_path.read_text(encoding='utf-8'))
    if report['status']!='complete' or report['protocol_sha256']!=sha256(source/'protocol.json'):
        raise ValueError('not completed or changed protocol')
    for p,d in protocol['pins'].items():
        if sha256(p)!=d:raise ValueError('pinned source changed')
    if source_pins()!=protocol['mainline_pins']:raise ValueError('E mainline drift')
    output=ROOT/'artifacts/mendeley_socket_appearance_independent_audit_20261005'
    if output.exists():raise FileExistsError('preserve old audit')
    output.mkdir(exist_ok=False)
    audit_pins={str(p):sha256(p) for p in [Path(__file__),Path(__file__).with_name('socket_appearance.py'),
        source/'report.json',source/'protocol.json',model_path]}
    begun=time.perf_counter();features={'fit':[],'calibration':[],'inspection':[]};paths={key:set() for key in features}
    row_ids=set();replayed=[]
    for row in report['prepared']:
        identity=row['id'];phase=row['phase'];path=row['path']
        if identity in row_ids or path in paths[phase]:raise ValueError('duplicate prepared record')
        row_ids.add(identity);paths[phase].add(path)
        if sha256(path)!=row['input_sha256']:raise ValueError('source image drift')
        if not row['appearance_prepared']:raise ValueError('unexpected missing appearance input')
        if not row['registration']['alignment_quality']['reliable'] or not row['pose']['localization_proposal_supported']:
            raise ValueError('appearance built on unqualified geometry')
        if not all(row['pose']['gates'].values()):raise ValueError('stored pose proposal silently accepts failed gate')
        patch_path=source/(identity+'_socket.png')
        with Image.open(path) as opened:original=np.asarray(opened.convert('RGB'))
        h=np.array([[1,0,-1600],[0,1,-1000],[0,0,1]])@np.asarray(row['pose']['inspection_to_reference_local'])
        valid=cv2.warpPerspective(np.ones(original.shape[:2],np.uint8),h,(100,50),flags=cv2.INTER_NEAREST)
        if not np.all(valid):raise ValueError('invalid interpolation support')
        fresh=cv2.warpPerspective(original,h,(100,50),flags=cv2.INTER_LINEAR)
        with Image.open(patch_path) as opened:saved=np.asarray(opened.convert('RGB'))
        if not np.array_equal(fresh,saved):raise ValueError('patch differs from current original pixels and transform')
        f=descriptor(fresh)
        if not np.array_equal(f,np.asarray(row['descriptor'])):raise ValueError('descriptor drift')
        features[phase].append(f)
        if phase=='inspection':replayed.append({'id':identity,'descriptor':f,'source_sha256':row['input_sha256']})
    for phase,expected in [('fit',80),('calibration',40),('inspection',30)]:
        if len(features[phase])!=expected or paths[phase]!=set(protocol[phase+'_paths' if phase!='fit' else 'fit_paths']):
            raise ValueError('all input inventory or fit/calibration split differs')
    hashes={phase:{protocol['pins'][p] for p in group} for phase,group in paths.items()}
    if any(hashes[a]&hashes[b] for a,b in [('fit','calibration'),('fit','inspection'),('calibration','inspection')]):
        raise ValueError('image-byte leakage between fit/calibration/inspection')
    f=np.asarray(features['fit']);center=np.mean(f,axis=0);residual=f-center
    empirical=residual.T@residual/(len(f)-1)
    ridge=max(float(np.trace(empirical)/320),1e-6)
    covariance=.9*empirical+.1*ridge*np.eye(320);precision=np.linalg.inv(covariance)
    if not np.allclose(center,model['center'],rtol=0,atol=1e-12) or not np.allclose(precision,model['precision'],rtol=1e-8,atol=1e-8):
        raise ValueError('normal-only model does not replay')
    def scores(fs):
        diff=np.asarray(fs)-center
        return np.sqrt(np.maximum(0,np.einsum('ni,ij,nj->n',diff,precision,diff)))
    calib=scores(features['calibration']);inspection=scores(features['inspection'])
    if not np.allclose(calib,model['calibration_scores'],rtol=1e-9,atol=1e-9):raise ValueError('calibration scores differ')
    for row,value in zip(replayed,inspection):
        original=next(r for r in report['cases'] if r['id']==row['id'])
        p=float((1+np.count_nonzero(calib>=value))/(len(calib)+1));candidate=p<=.05
        expected=original['appearance_evidence']
        if not np.isclose(value,expected['appearance_score'],rtol=1e-8,atol=1e-8) or p!=expected['normal_calibration_tail_rank']:
            raise ValueError('appearance rank does not independently replay')
        if candidate!=expected['appearance_difference_candidate'] or expected['plug_absent_confirmed']:
            raise ValueError('appearance hypothesis wrongly promoted or differs')
        row.pop('descriptor');row.update(appearance_score=float(value),calibration_tail_rank=p,appearance_candidate=bool(candidate))
    if report['appearance_difference_candidates']!=sum(r['appearance_candidate'] for r in replayed):raise ValueError('candidate count differs')
    if any(sha256(p)!=d for p,d in audit_pins.items()) or source_pins()!=protocol['mainline_pins']:raise ValueError('source changed during audit')
    result={'status':'PASS','seconds':time.perf_counter()-begun,'freshly_reconstructed_original_image_patches':150,
        'fit80_calibration40_inspection30_disjoint_byte_hashes':True,'normal_model_replayed':True,
        'cases':replayed,'appearance_candidates':sum(r['appearance_candidate'] for r in replayed),
        'new_confirmed_connections':0,'confirmed_disconnections':0,'deployed':False,
        'not_fault_accuracy_or_semantic_identity':True,'source_pins':audit_pins}
    save(output/'report.json',result)
    print(json.dumps({k:result[k] for k in ['status','seconds','freshly_reconstructed_original_image_patches',
        'appearance_candidates','new_confirmed_connections']}))


if __name__=='__main__':main()
