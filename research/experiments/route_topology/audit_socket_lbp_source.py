"""Independent score/rank/retention replay using explicit stored features."""
import json
from pathlib import Path
import numpy as np
from core import sha256
from run_audit import source_pins
from run_review import save

ROOT=Path(__file__).resolve().parents[2]

def main():
    folder=ROOT/'artifacts/mendeley_socket_lbp_source_20261008'
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    protocol=read(folder/'protocol.json');result=read(folder/'report.json')
    for p,digest in protocol['pins'].items():
        if sha256(p)!=digest:raise ValueError('fixed input/code drift')
    if source_pins()!=protocol['mainline_pins']:raise ValueError('E drift')
    model=read(folder/'model.json');records=read(folder/'features.json')['records']
    features={r['id']:r for r in records}
    if len(features)!=197 or set(protocol['fit_ids']) & set(protocol['calibration_ids']):
        raise ValueError('source split changed')
    def probability(f):
        z=(np.array(f)-np.array(model['center']))/np.array(model['scale'])
        return float(1/(1+np.exp(-np.clip(z@np.array(model['weights'])+model['bias'],-40,40))))
    labels={r['id']:r['visual_label'] for r in result['cases']}
    scores={identity:probability(features[identity]['feature']) for identity in protocol['calibration_ids']}
    rows=[];gains=[];wrong=0;resolved=0
    for row in result['cases']:
        expected=[]
        for vector,reported in zip(features[row['id']]['probe_features'],row['probes']):
            p=probability(vector)
            ranks={}
            for k in [0,1]:
                calibration=[scores[i] if k==0 else 1-scores[i] for i in scores if i!=row['id'] and labels[i]==k]
                score=p if k==0 else 1-p
                ranks[str(k)]=(1+sum(s>=score for s in calibration))/(1+len(calibration))
            choices=[k for k in [0,1] if ranks[str(k)]>.05]
            predicted=choices[0] if len(choices)==1 else None
            if abs(p-reported['probability'])>1e-12 or ranks!=reported['class_tail_ranks'] or predicted!=reported['visual_label_candidate']:
                raise ValueError('head or class-tail differs')
            expected.append(predicted)
        if len(expected)!=9:raise ValueError('probe count differs')
        final=row['old_candidate']
        if final is None and expected[0] is not None and len(set(expected))==1:final=expected[0]
        if final!=row['prediction']['visual_label_candidate']:raise ValueError('retention/unanimity differs')
        if final is not None:resolved+=1;wrong+=int(final!=row['visual_label'])
        if row['old_candidate'] is None and final==row['visual_label']:gains.append(row['id'])
        rows.append(dict(id=row['id'],final=final))
    if (wrong!=result['source_gate']['wrong_singletons'] or resolved!=result['source_gate']['singleton_count']
            or gains!=result['new_source_support']):raise ValueError('counts differ')
    out=ROOT/'artifacts/mendeley_socket_lbp_source_audit_20261008';out.mkdir(exist_ok=False)
    report=dict(status='PASS',scope='independent_score_rank_retention_replay_shared_stored_features',
        source_report_sha256=sha256(folder/'report.json'),sources=80,probes=720,
        new_source_support=gains,resolved=resolved,wrong=wrong,fresh_pixel_descriptor_recomputed=False,
        not_independent_field_validation=True,automatic_topology_new_hits=0,deployed=False)
    save(out/'report.json',report);print(json.dumps(report))

if __name__=='__main__':main()
