"""Why two fixed training-only context studies rejected every proposal; no retuning."""
import json,sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
from core_port_precision_policy import iou
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    records=[]
    for experiment,key in (('teacher_context_student_20261003','teacher_context'),('student_microcontext_20261003','microcontext')):
        report=load(ROOT/'artifacts'/experiment/'train/report.json')
        for case in report['cases']:
            if not case['seeds']:continue
            data=load(ROOT/'artifacts'/experiment/'train'/(Path(case['image']).stem+'_predictions.json'))
            for entry in data[key]:
                seed=entry['proposal'];views=[]
                for view in entry['views']:
                    same=[p for p in view if p['class_id']==seed['class_id']]
                    matching=[p for p in same if iou(p['box_xyxy'],seed['box_xyxy'])>=.5]
                    strong=[p for p in same if p['confidence']>.75]
                    views.append(dict(maximum_same_class_iou=max((iou(p['box_xyxy'],seed['box_xyxy']) for p in same),default=0),
                        maximum_matching_score=max((p['confidence'] for p in matching),default=None),
                        maximum_strong_box_seed_iou=max((iou(p['box_xyxy'],seed['box_xyxy']) for p in strong),default=0),
                        matching_strong_count=sum(p['confidence']>.75 for p in matching)))
                records.append(dict(experiment=experiment,image=case['image'],seed_score=seed['confidence'],seed_class=seed['class_id'],views=views))
    out=ROOT/'artifacts/context_confirmation_diagnosis_20261003'
    if out.exists():raise FileExistsError('Fresh output required')
    out.mkdir();result=dict(status='complete',training_only=True,records=records,no_threshold_changes=True)
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
