"""Training-only error decomposition; no production rule/score changes."""
import json,sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];REPO=Path('E:/PythonProject10');sys.path.insert(0,str(REPO))
from audit_port_multiscale_acceptance import matches,overlap
from inspection_agent.optional_port_crop_review import sha
DATA=REPO/'data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    out=ROOT/'artifacts/port_training_recall_ceiling_20261003'
    if out.exists():raise FileExistsError('Fresh output required')
    prior=load(ROOT/'artifacts/teacher_student_port_20261003/train/report.json');records=[];totals=dict(targets=0,selected=0,budget_oracle=0,raw_oracle=0,raw_union_oracle=0)
    for item in prior['cases']:
        name=item['image'];stem=Path(name).stem
        old=load(ROOT/'artifacts/core_port_recheck_20261002/train'/(stem+'_zoom_predictions.json'))
        new=load(ROOT/'artifacts/port_training_multiscale_20261002/evaluation/train'/(stem+'_zoom_predictions.json'))
        evaluated=load(ROOT/'artifacts/teacher_student_port_20261003/train'/(stem+'_evaluation.json'))
        assert sha(DATA/'images/train01'/name)==old['source_sha256']==new['source_sha256']
        label=DATA/'labels/train01'/(stem+'.txt');assert sha(label)==evaluated['label_sha256']
        h,w=old['predictions']['source_shape'];targets=[]
        for line in label.read_text(encoding='utf-8').splitlines():
            cls,cx,cy,bw,bh=map(float,line.split())
            if cls in (3,4):targets.append(dict(class_id=int(cls)-3,box=[(cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h]))
        selected=evaluated['selected']['cross_model_supported']['all_predictions'];matched=matches(selected,targets)[0]
        raw_old=old['predictions']['merged_predictions'];raw_new=new['predictions']['merged_predictions']
        raw_union=[p for p in raw_old+raw_new if p['confidence']>.25]
        union_matches=matches(raw_union,targets)[0];old_matches=matches([p for p in raw_old if p['confidence']>.25],targets)[0]
        misses=[]
        for index,target in enumerate(targets):
            if index in matched:continue
            same=[p for p in raw_old+raw_new if p['class_id']==target['class_id'] and overlap(p['box_xyxy'],target['box'])>=.5]
            best=max(same,key=lambda p:p['confidence']) if same else None
            reason='no_retained_raw_box_at_iou05' if not best else 'low_score' if best['confidence']<=.5 else 'strong_raw_excluded_by_budget_or_confirmation'
            misses.append(dict(target_index=index,reason=reason,best_same_class_score=best['confidence'] if best else None))
        values=dict(targets=len(targets),selected=len(matched),budget_oracle=min(10,len(targets)),raw_oracle=len(old_matches),raw_union_oracle=len(union_matches))
        for key,value in values.items():totals[key]+=value
        records.append(dict(image=name,**values,misses=misses))
    out.mkdir();result=dict(status='complete',training_only=True,summary=totals,cases=records,
        oracle_not_achieved_accuracy=True,thresholds_unchanged=True,no_new_inference=True,no_budget_increase=True)
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    counts={}
    for case in records:
        for miss in case['misses']:counts[miss['reason']]=counts.get(miss['reason'],0)+1
    print(json.dumps(dict(summary=totals,miss_reasons=counts),indent=2))
if __name__=='__main__':main()
