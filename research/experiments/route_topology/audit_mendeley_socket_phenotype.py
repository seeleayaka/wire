"""Independent original-pixel and optimisation replay; no inspection images."""
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
    source=ROOT/'artifacts/mendeley_socket_source_phenotype_20261005'
    protocol=json.loads((source/'protocol.json').read_text(encoding='utf-8'))
    report=json.loads((source/'report.json').read_text(encoding='utf-8'))
    model=json.loads((source/'model.json').read_text(encoding='utf-8'))
    label_path=ROOT/'artifacts/mendeley_source_socket_visual_labels_20261005/labels.json'
    labels=json.loads(label_path.read_text(encoding='utf-8'));inventory={r['id']:r for r in labels['rows']}
    if report['protocol_sha256']!=sha256(source/'protocol.json'):raise ValueError('protocol drift')
    if any(sha256(p)!=d for p,d in protocol['pins'].items()) or source_pins()!=protocol['mainline_pins']:
        raise ValueError('code/input/E changed')
    for p,d in labels['reviewed_contact_sheet_pins'].items():
        if sha256(p)!=d:raise ValueError('reviewed source sheet drift')
    begun=time.perf_counter();features={};hashes={'fit':set(),'calibration':set()}
    for row in report['prepared']:
        original=inventory[row['id']]
        if sha256(original['source_path'])!=row['source_sha256']:raise ValueError('source drift')
        if not all(original['pose']['gates'].values()):raise ValueError('source spatial gate failed')
        if row['id'] in features:raise ValueError('duplicate source row')
        hashes[row['phase']].add(row['source_sha256'])
        with Image.open(original['source_path']) as im:rgb=np.asarray(im.convert('RGB'))
        h=np.array([[1,0,-1600],[0,1,-1000],[0,0,1]])@np.array(original['pose']['inspection_to_reference_local'])
        valid=cv2.warpPerspective(np.ones(rgb.shape[:2],np.uint8),h,(100,50),flags=cv2.INTER_NEAREST)
        if not valid.all():raise ValueError('unsupported interpolation')
        patch=cv2.warpPerspective(rgb,h,(100,50),flags=cv2.INTER_LINEAR)
        if sha256(original['patch_path'])!=original['patch_sha256']:raise ValueError('review patch drift')
        with Image.open(original['patch_path']) as im:saved=np.asarray(im.convert('RGB'))
        if not np.array_equal(patch,saved):raise ValueError('pixel replay differs')
        f=descriptor(patch)
        if not np.array_equal(f,row['descriptor']):raise ValueError('feature replay differs')
        features[row['id']]=f
    if hashes['fit']&hashes['calibration'] or [len(hashes[k]) for k in ['fit','calibration']]!=[117,80]:
        raise ValueError('source split leakage')
    x=np.array([features[k] for k in protocol['fit_ids']]);y=np.array([inventory[k]['visual_label'] for k in protocol['fit_ids']])
    center=np.average(x,axis=0);scale=np.maximum(np.sqrt(np.average((x-center)**2,axis=0)),.01)
    z=(x-center)/scale;w=np.zeros(320);b=0.
    class_weight=np.where(y==0,len(y)/(2*np.count_nonzero(y==0)),len(y)/(2*np.count_nonzero(y==1)))
    for _ in range(1000):
        p=1/(1+np.exp(-np.clip(z@w+b,-40,40)));residual=(p-y)*class_weight
        gradient=np.sum(z*residual[:,None],axis=0)/len(y)+.02*w
        w=w-.05*gradient;b=b-.05*np.average(residual)
    for key,value in [('center',center),('scale',scale),('weights',w),('bias',b)]:
        if not np.allclose(value,model[key],rtol=1e-9,atol=1e-10):raise ValueError('optimisation replay differs')
    def prob(identity):
        logit=np.dot((features[identity]-center)/scale,w)+b
        return float(1/(1+np.exp(-np.clip(logit,-40,40))))
    scores={k:[prob(i) if k==0 else 1-prob(i) for i in protocol['calibration_ids'] if inventory[i]['visual_label']==k] for k in [0,1]}
    for k in [0,1]:
        if not np.allclose(scores[k],model['calibration'][str(k)],atol=1e-10):raise ValueError('calibration replay differs')
    wrong=0;singletons=0
    for row in report['calibration_results']:
        p=prob(row['id']);rank={k:(1+sum(v>=candidate for v in scores[k]))/(1+len(scores[k])) for k,candidate in [(0,p),(1,1-p)]}
        admitted=[k for k in [0,1] if rank[k]>.05];pred=admitted[0] if len(admitted)==1 else None
        if pred!=row['prediction']['visual_label_candidate']:raise ValueError('prediction replay differs')
        singletons+=pred is not None;wrong+=pred is not None and pred!=row['visual_label']
    if wrong!=report['source_gate']['wrong_singletons'] or singletons!=report['source_gate']['singleton_count']:
        raise ValueError('source gate replay differs')
    if source_pins()!=protocol['mainline_pins']:raise ValueError('E changed during replay')
    output=ROOT/'artifacts/mendeley_socket_source_phenotype_audit_20261005'
    output.mkdir(exist_ok=False)
    result={'status':'PASS','source_gate_passed':report['source_gate']['passed'],
        'reconstructed_source_originals':197,'optimisation_independently_replayed':True,
        'source_split_byte_disjoint':True,'wrong_singletons':int(wrong),'singleton_count':int(singletons),
        'inspection_inputs_or_labels_read':False,'not_field_accuracy':True,'new_confirmed_connections':0,
        'seconds':time.perf_counter()-begun,'pins':{str(p):sha256(p) for p in [Path(__file__),source/'report.json',source/'model.json',source/'protocol.json']}}
    save(output/'report.json',result);print(json.dumps(result))


if __name__=='__main__':main()
