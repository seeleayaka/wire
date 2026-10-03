"""Locked policy replay on existing fresh 960 predictions, no model retraining."""
import sys,json
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/core_port_budget_test01_20261002'
from core_port_budget_policy import bounded_expand
def main():
    from core_port_resolution_ab_20261002 import score,ranking,DATA,WEIGHT,sha
    inner=json.loads((ROOT/'artifacts/core_port_budget_policy_20261002/report.json').read_text(encoding='utf-8'))
    a=inner['summary']['960']['top1'];b=inner['summary']['960']['bounded_expand']
    if not (b['tp']>a['tp'] and b['unmatched']<=a['unmatched']):
        print('INNER VALIDATION REJECTED; NO TEST REPLAY');return
    if OUT.exists():raise FileExistsError('Fresh output required')
    OUT.mkdir()
    base=ROOT/'artifacts/port_bridge_fix_20261002/independent_test01'
    manifest=json.loads((base/'manifest.json').read_text(encoding='utf-8'))
    assert sha(WEIGHT)==manifest['weight_sha256']
    summary={};cases=[]
    for source in manifest['images']:
        name=source['name'];path=DATA/'images/test01'/name;assert sha(path)==source['sha256']
        cached=base/(Path(name).stem+'_predictions.json')
        case=json.loads(cached.read_text(encoding='utf-8'))
        rows=case['predictions']['merged_predictions']
        chosen={'top1':ranking(rows,1),'bounded_expand':bounded_expand(rows)}
        targets=[]
        for line in (DATA/'labels/test01'/(Path(name).stem+'.txt')).read_text(encoding='utf-8').splitlines():
            cls,cx,cy,w,h=map(float,line.split())
            if cls in (3,4):targets.append(dict(class_id=int(cls)-3,box=[(cx-w/2)*3648,(cy-h/2)*2736,(cx+w/2)*3648,(cy+h/2)*2736]))
        result={}
        for stage,predictions in chosen.items():
            metrics=score(predictions,targets);result[stage]=metrics
            aggregate=summary.setdefault(stage,{k:0 for k in metrics})
            for key,value in metrics.items():aggregate[key]+=value
        cases.append(dict(image=name,metrics=result,cached_prediction_sha256=sha(cached)))
    report=dict(status='complete',summary=summary,cases=cases,new_inference=False,
        predictor='same frozen960 model, full cached30 source predictions from20261002',
        already_inspected_test_set=True,not_independent_unseen_accuracy=True,formal_policy_changed=False)
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
