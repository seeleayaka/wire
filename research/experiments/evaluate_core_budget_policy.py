"""Freeze policy before reading labels, evaluate inner holdout first."""
import sys,json
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/core_port_resolution_ab_20261002'
OUT=ROOT/'artifacts/core_port_budget_policy_20261002'
from core_port_budget_policy import bounded_expand
def freeze():
    if OUT.exists():raise FileExistsError('Do not overwrite protocol')
    OUT.mkdir()
    protocol=dict(base_threshold=.25,extra_threshold=.5,maximum=5,baseline_maximum=1,
        selection='Keep original top1; add only remaining scores >0.5, max5 total',
        budget_change_explicit=True,changes_formal_policy=False,
        warning='Scores are not calibrated probabilities; gain must include unmatched and burden counts')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf-8')
    print('BUDGET POLICY FROZEN BEFORE INNER LABEL EVALUATION')
def evaluate():
    from core_port_resolution_ab_20261002 import score,ranking,DATA
    protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    assert protocol['extra_threshold']==.5 and protocol['maximum']==5
    source=json.loads((BASE/'report.json').read_text(encoding='utf-8'));summary={};cases=[]
    for case in source['cases']:
        target=[]
        for line in (DATA/'labels/train01'/(Path(case['image']).stem+'.txt')).read_text(encoding='utf-8').splitlines():
            cls,cx,cy,w,h=map(float,line.split())
            if cls in (3,4):target.append(dict(class_id=int(cls)-3,box=[(cx-w/2)*3648,(cy-h/2)*2736,(cx+w/2)*3648,(cy+h/2)*2736]))
        raw=json.loads((BASE/(Path(case['image']).stem+'_predictions.json')).read_text(encoding='utf-8'))
        result={}
        for size in ('960','1280'):
            predictions=raw['predictions'][size]['merged_predictions']
            stages={'top1':ranking(predictions,1),'bounded_expand':bounded_expand(predictions)}
            result[size]={}
            for stage,rows in stages.items():
                metrics=score(rows,target);result[size][stage]=metrics
                aggregate=summary.setdefault(size,{}).setdefault(stage,{k:0 for k in metrics})
                for key,value in metrics.items():aggregate[key]+=value
        cases.append(dict(image=case['image'],metrics=result))
    report=dict(status='complete',inner_validation_only=True,summary=summary,cases=cases,
        test_or_outer_validation_read=False,formal_policy_changed=False)
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':freeze() if '--freeze' in sys.argv else evaluate()
