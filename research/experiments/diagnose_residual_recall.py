"""Current training-only recall decomposition, not a rule change or achieved oracle."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];REPO=Path('E:/PythonProject10');sys.path.insert(0,str(REPO));sys.dont_write_bytecode=True
from inspection_agent.optional_port_crop_review import sha
from audit_port_multiscale_acceptance import metric,matches,overlap
DATA=REPO/'data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults'
BASE=ROOT/'artifacts/port_residual_feature_support_20261003'
OUT=ROOT/'artifacts/residual_training_recall_diagnosis_20261003'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    if OUT.exists():raise FileExistsError('Preserve earlier diagnosis')
    source_report=load(BASE/'train/report.json');records=[];counts={};totals=dict(targets=0,selected=0,budget_oracle=0,raw_union_oracle=0)
    for entry in source_report['cases']:
        name=entry['image'];stem=Path(name).stem
        teacher=load(ROOT/'artifacts/core_port_recheck_20261002/train'/(stem+'_zoom_predictions.json'))
        student=load(ROOT/'artifacts/port_training_multiscale_20261002/evaluation/train'/(stem+'_zoom_predictions.json'))
        current=load(BASE/'train'/(stem+'_predictions.json'));feature=current['feature'];selected=current['trial']['all_predictions']
        source=DATA/'images/train01'/name;assert sha(source)==teacher['source_sha256']==student['source_sha256']==feature['source_sha256']
        label=DATA/'labels/train01'/(stem+'.txt');assert sha(label)==entry['label_sha256']
        h,w=feature['predictions']['source_shape'];targets=[]
        for line in label.read_text(encoding='utf-8').splitlines():
            cls,cx,cy,bw,bh=map(float,line.split())
            if cls in (3,4):targets.append(dict(class_id=int(cls)-3,box=[(cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h]))
        score=metric(selected,targets);assert score==entry['metrics']['trial'];hits=matches(selected,targets)[0]
        raw=[p for model in (teacher,student,feature) for p in model['predictions']['merged_predictions']]
        misses=[]
        for index,target in enumerate(targets):
            if index in hits:continue
            matching=[p for p in raw if p['class_id']==target['class_id'] and overlap(p['box_xyxy'],target['box'])>=.5]
            best=max(matching,key=lambda p:p['confidence']) if matching else None
            reason='no_raw_geometry' if best is None else 'low_score' if best['confidence']<=.5 else 'strong_raw_capacity_or_confirmation'
            counts[reason]=counts.get(reason,0)+1
            misses.append(dict(target=index,class_id=target['class_id'],reason=reason,best_score=best['confidence'] if best else None,
                width=target['box'][2]-target['box'][0],height=target['box'][3]-target['box'][1],budget_full=len(selected)>=len(current['trial']['primary'])+5))
        values=dict(targets=len(targets),selected=score['tp'],budget_oracle=min(10,len(targets)),
            raw_union_oracle=len(matches([p for p in raw if p['confidence']>.25],targets)[0]))
        for k,v in values.items():totals[k]+=v
        records.append(dict(image=name,**values,misses=misses))
    assert totals['targets']==98 and totals['selected']==68
    OUT.mkdir();report=dict(status='complete',summary=totals,miss_reasons=counts,cases=records,
        training_only=True,no_new_inference=True,oracle_not_achieved_accuracy=True,production_changed=False)
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(summary=totals,miss_reasons=counts)),flush=True)
if __name__=='__main__':main()
