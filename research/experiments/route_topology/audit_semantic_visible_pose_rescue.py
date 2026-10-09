"""Independent recorded-probability rank and asymmetric decision replay."""
import json
from pathlib import Path
from core import sha256
from run_prompt_contrast import save
from run_audit import source_pins
ROOT=Path(__file__).resolve().parents[2]

def main():
    source=ROOT/'artifacts/mendeley_semantic_visible_pose_rescue_20261006'
    rp=source/'report.json';r=json.loads(rp.read_text(encoding='utf-8'))
    mp=ROOT/'artifacts/mendeley_component_semantic_source_v2_20261006/model.json'
    model=json.loads(mp.read_text(encoding='utf-8'))
    bp=ROOT/'artifacts/mendeley_contact_translation_nuisance_20261006/report.json'
    baseline={c['id']:c for c in json.loads(bp.read_text(encoding='utf-8'))['cases']}
    pins=source_pins();gains=[];wrong=0;resolved=0;decisions=0
    assert r['status']=='complete' and len(r['cases'])==len(baseline)==80
    for row in r['cases']:
        i=row['id'];assert row['old_candidate']==baseline[i]['prediction']['visual_label_candidate']
        labels=[];assert len(row['semantic_probes'])==9
        for probe in row['semantic_probes']:
            p=probe['prediction'];prob=p['probability'];admitted=[]
            for k in (0,1):
                scores=[c['score'] for c in model['calibration'][str(k)] if c['id']!=i]
                query=prob if k==0 else 1-prob
                rank=(1+sum(s>=query for s in scores))/(1+len(scores))
                assert len(scores)>=19 and abs(rank-p['class_tail_ranks'][str(k)])<1e-12
                if rank>.05:admitted.append(k)
            label=admitted[0] if len(admitted)==1 else None
            assert label==p['visual_label_candidate'];labels.append(label);decisions+=1
        stable=labels==[1]*9;assert stable==row['stable_visible_body_proposal']
        previous=row['old_candidate'];final=previous if previous is not None else (1 if stable else None)
        assert final==row['prediction']['visual_label_candidate']
        if final is not None:resolved+=1;wrong+=final!=row['visual_label']
        if previous is None and final==row['visual_label']:gains.append(i)
    assert resolved==r['source_gate']['singleton_count'] and wrong==r['source_gate']['wrong_singletons']
    assert gains==r['gains'] and r['strict_net_source_gain']==(wrong==0 and bool(gains) and resolved>=60)
    assert source_pins()==pins
    out=ROOT/'artifacts/mendeley_semantic_visible_pose_rescue_audit_20261006';out.mkdir(exist_ok=False)
    save(out/'report.json',dict(status='PASS',source_report_sha256=sha256(rp),model_sha256=sha256(mp),
        rank_decisions_replayed=decisions,resolved=resolved,wrong=wrong,gains=gains,
        scope='independent rank and decision replay; NOT independently extracted descriptors or model fit',
        production_unchanged=True,deployed=False))
    print(json.dumps(dict(status='PASS',resolved=resolved,wrong=wrong,gains=gains)))
if __name__=='__main__':main()
