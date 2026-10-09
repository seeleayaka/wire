"""Independent rank/three-head/outer-offset/old-preservation decision replay.

Does not independently refit heads or recompute the image descriptors.
"""
import json
from pathlib import Path
from core import sha256
from run_prompt_contrast import save
from run_audit import source_pins

ROOT=Path(__file__).resolve().parents[2]

def main():
    source=ROOT/'artifacts/mendeley_mean_pose_source_20261006'
    out=ROOT/'artifacts/mendeley_mean_pose_decision_audit_20261006'
    report_path=source/'report.json';report=json.loads(report_path.read_text(encoding='utf-8'))
    assert report['status']=='complete'
    model=json.loads((source/'model.json').read_text(encoding='utf-8'))
    baseline_path=ROOT/'artifacts/mendeley_contact_translation_nuisance_20261006/report.json'
    baseline={r['id']:r for r in json.loads(baseline_path.read_text(encoding='utf-8'))['cases']}
    pins=source_pins();gains=[];wrong=0;resolved=0;probabilities=0
    for row in report['cases']:
        i=row['id'];assert row['old_candidate']==baseline[i]['prediction']['visual_label_candidate']
        outer=[]
        for probe in row['outer_probes']:
            per_head=[]
            for name,result in probe['feature_view_predictions'].items():
                p=result['probability'];assert 0<=p<=1
                scores=model[name]['calibration'];labels=[]
                for k in (0,1):
                    remaining=[s['score'] for s in scores[str(k)] if s['id']!=i]
                    assert len(remaining)>=19
                    query=p if k==0 else 1-p
                    rank=(1+sum(s>=query for s in remaining))/(1+len(remaining))
                    assert abs(rank-result['class_tail_ranks'][str(k)])<1e-12
                    if rank>.05:labels.append(k)
                candidate=labels[0] if len(labels)==1 else None
                assert candidate==result['visual_label_candidate']
                per_head.append(candidate);probabilities+=1
            unique=set(per_head)
            label=next(iter(unique)) if len(unique)==1 else None
            assert label==probe['visual_label_candidate'];outer.append(label)
        assert len(outer)==9
        unique=set(outer);proposal=next(iter(unique)) if len(unique)==1 else None
        assert proposal==row['proposed']
        final=row['old_candidate'] if row['old_candidate'] is not None else proposal
        assert final==row['prediction']['visual_label_candidate']
        if final is not None:
            resolved+=1;wrong+=final!=row['visual_label']
        if row['old_candidate'] is None and final==row['visual_label']:gains.append(i)
    assert len(report['cases'])==len(baseline)==80
    assert resolved==report['source_gate']['singleton_count']
    assert wrong==report['source_gate']['wrong_singletons']
    assert gains==report['gains']
    assert report['strict_net_source_gain']==(wrong==0 and resolved>=60 and bool(gains))
    assert source_pins()==pins
    out.mkdir(exist_ok=False)
    save(out/'report.json',dict(status='PASS',cases=80,head_decisions_replayed=probabilities,
        source_report_sha256=sha256(report_path),resolved=resolved,wrong=wrong,gains=gains,
        scope='rank and decision replay, NOT independent descriptor extraction/refitting',
        production_unchanged=True,deployed=False))
    print(json.dumps(dict(resolved=resolved,wrong=wrong,gains=gains)))

if __name__=='__main__':main()
