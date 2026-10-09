"""Replay each fixed feature model and all source decisions independently."""
import json
from pathlib import Path
import numpy as np
from core import sha256
from run_audit import source_pins
from run_review import save

ROOT=Path(__file__).resolve().parents[2]


def main():
    source=ROOT/'artifacts/mendeley_socket_phenotype_agreement_20261005'
    protocol=json.loads((source/'protocol.json').read_text(encoding='utf-8'))
    report=json.loads((source/'report.json').read_text(encoding='utf-8'))
    models=json.loads((source/'model.json').read_text(encoding='utf-8'))
    old=json.loads((ROOT/'artifacts/mendeley_socket_source_phenotype_20261005/report.json').read_text(encoding='utf-8'))
    if report['protocol_sha256']!=sha256(source/'protocol.json') or any(sha256(p)!=d for p,d in protocol['pins'].items()):raise ValueError('source drift')
    if source_pins()!=protocol['mainline_pins']:raise ValueError('E drift')
    inventory={r['id']:r for r in old['prepared']};fit_ids=protocol['source_protocol']['fit_ids'];calib_ids=protocol['source_protocol']['calibration_ids']
    replay={};y=np.array([inventory[k]['visual_label'] for k in fit_ids])
    for view in ['combined','color','edge']:
        indices=np.ones(320,dtype=bool) if view=='combined' else np.arange(320)%10<6
        if view=='edge':indices=~indices
        def features(identity):
            value=np.asarray(inventory[identity]['descriptor']).copy();value[~indices]=0;return value
        x=np.array([features(k) for k in fit_ids]);center=x.mean(0);scale=np.maximum(x.std(0),.01)
        z=(x-center)/scale;w=np.zeros(320);b=0.
        balance=np.array([len(y)/(2*np.count_nonzero(y==k)) for k in y])
        for _ in range(1000):
            logits=np.clip(z@w+b,-40,40);residual=(1/(1+np.exp(-logits))-y)*balance
            w-=.05*(np.sum(z*residual[:,None],0)/len(y)+.02*w);b-=.05*residual.mean()
        for key,value in [('center',center),('scale',scale),('weights',w),('bias',b)]:
            if not np.allclose(value,models[view][key],rtol=1e-9,atol=1e-10):raise ValueError('model replay differs')
        probabilities={k:float(1/(1+np.exp(-np.clip(np.dot((features(k)-center)/scale,w)+b,-40,40)))) for k in calib_ids}
        scores={label:[p if label==0 else 1-p for k,p in probabilities.items() if inventory[k]['visual_label']==label] for label in [0,1]}
        for label in [0,1]:
            if not np.allclose(scores[label],models[view]['calibration'][str(label)],atol=1e-10):raise ValueError('class calibration drift')
        replay[view]={}
        for identity,p in probabilities.items():
            rank={k:(1+sum(v>=s for v in scores[k]))/(1+len(scores[k])) for k,s in [(0,p),(1,1-p)]}
            admitted=[k for k in [0,1] if rank[k]>.05];replay[view][identity]=admitted[0] if len(admitted)==1 else None
    wrong=0;singletons=0
    for r in report['calibration_results']:
        values=[replay[v][r['id']] for v in ['combined','color','edge']]
        predicted=values[0] if values[0] is not None and len(set(values))==1 else None
        if predicted!=r['prediction']['visual_label_candidate']:raise ValueError('agreement replay differs')
        singletons+=predicted is not None;wrong+=predicted is not None and predicted!=r['visual_label']
    if report['source_gate']['wrong_singletons']!=wrong or report['source_gate']['singleton_count']!=singletons:raise ValueError('gate count drift')
    output=ROOT/'artifacts/mendeley_socket_phenotype_agreement_audit_20261005';output.mkdir(exist_ok=False)
    result={'status':'PASS','models_independently_replayed':3,'source_gate_passed':wrong==0 and singletons/80>=.75,
        'source_singletons':int(singletons),'wrong_source_singletons':int(wrong),
        'not_independent_accuracy_test':True,'inspection_inputs_or_labels_read':False,
        'new_confirmed_connections':0,'reference_review_confirmed':False,
        'pins':{str(p):sha256(p) for p in [Path(__file__),source/'model.json',source/'report.json',source/'protocol.json']}}
    save(output/'report.json',result);print(json.dumps(result))


if __name__=='__main__':main()
