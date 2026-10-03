"""Explain rejected training-only semantic support without changing thresholds."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];REPO=Path('E:/PythonProject10');sys.path.insert(0,str(REPO))
from audit_port_multiscale_acceptance import matches,overlap
from inspection_agent.optional_port_crop_review import sha
from port_semantic_support_policy import proposals
DATA=REPO/'data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults'
SOURCE=ROOT/'artifacts/port_semantic_source_support_20261003'
OUT=ROOT/'artifacts/semantic_source_miss_diagnosis_20261003'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    if OUT.exists():raise FileExistsError('Preserve diagnostics')
    report=load(SOURCE/'report.json');assert not report['qualifies']
    counts={};records=[]
    for row in report['cases']:
        if not row['image'].startswith('disconnected_'):continue
        name=row['image'];stem=Path(name).stem;case=load(SOURCE/(stem+'_predictions.json'))
        teacher=load(ROOT/'artifacts/core_port_recheck_20261002/train'/(stem+'_zoom_predictions.json'))
        student=load(ROOT/'artifacts/port_training_multiscale_20261002/evaluation/train'/(stem+'_zoom_predictions.json'))
        feature=load(ROOT/'artifacts/port_residual_feature_support_20261003/train'/(stem+'_predictions.json'))['feature']
        high=load(ROOT/'artifacts/port_high_resolution_support_20261003/train'/(stem+'_predictions.json'))['alternative']
        eligible=proposals(teacher,[teacher,student,feature,high]);h,w=teacher['predictions']['source_shape']
        label=DATA/'labels/train01'/(stem+'.txt');assert sha(label)==row['label_sha256'];targets=[]
        for line in label.read_text(encoding='utf-8').splitlines():
            cls,cx,cy,bw,bh=map(float,line.split())
            if cls in (3,4):targets.append(dict(class_id=int(cls)-3,box=[(cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h]))
        hits=matches(case['current']['all_predictions'],targets)[0];misses=[]
        full=len(case['current']['all_predictions'])>=len(case['current']['primary'])+5
        for index,target in enumerate(targets):
            if index in hits:continue
            eligible_matching=[p for p in eligible if p['class_id']==target['class_id'] and overlap(p['box_xyxy'],target['box'])>=.5]
            probabilities=[prob[target['class_id']+1] for p,prob in zip(case['candidates'],case['probabilities']) if
                p['class_id']==target['class_id'] and overlap(p['box_xyxy'],target['box'])>=.5]
            reason='budget_full' if full else 'no_eligible_geometry' if not eligible_matching else 'semantic_below_frozen_gate'
            counts[reason]=counts.get(reason,0)+1
            misses.append(dict(target=index,reason=reason,eligible=len(eligible_matching),best_semantic_probability=max(probabilities,default=None)))
        records.append(dict(image=name,misses=misses,eligible_candidates=len(eligible),classified_candidates=len(case['candidates'])))
    OUT.mkdir();(OUT/'report.json').write_text(json.dumps(dict(status='complete',counts=counts,cases=records,
        training_only=True,thresholds_not_changed=True,production_changed=False),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(counts=counts,cases=records)),flush=True)
if __name__=='__main__':main()
