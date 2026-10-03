"""Compare portable selector with the frozen experimental selector, all 110 cases."""
import json,sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];REPO=Path('E:/PythonProject10');sys.path.insert(0,str(REPO))
from inspection_agent.teacher_student_port_support import complementary_candidates,MANIFEST_SHA,MANIFEST_NAME,TEACHER_SHA,STUDENT_SHA,TEACHER_RELATIVE,STUDENT_RELATIVE
from inspection_agent.optional_port_crop_review import sha
from teacher_student_port_policy import merge

def load(p):return json.loads(p.read_text(encoding='utf-8'))
def canonical(rows):return [(p['class_id'],p['box_xyxy'],p['confidence']) for p in rows]
def main():
    out=ROOT/'artifacts/teacher_student_port_formal_20261003';out.mkdir(exist_ok=True)
    assert sha(REPO/'config'/MANIFEST_NAME)==MANIFEST_SHA
    assert sha(REPO/TEACHER_RELATIVE)==TEACHER_SHA and sha(REPO/STUDENT_RELATIVE)==STUDENT_SHA
    records=[]
    for mode in ('train','inner','outer'):
        for entry in load(ROOT/'artifacts/teacher_student_port_20261003'/mode/'report.json')['cases']:
            stem=Path(entry['image']).stem
            teacher=load(ROOT/'artifacts/core_port_recheck_20261002'/mode/(stem+'_zoom_predictions.json'))
            student=load(ROOT/'artifacts/port_training_multiscale_20261002/evaluation'/mode/(stem+'_zoom_predictions.json'))
            experiment=merge(teacher,student);formal=complementary_candidates(teacher,student)
            assert formal['fallback_reason'] is None,(mode,stem,formal['fallback_reason'])
            for key in ('primary','supplementary','zoom','student_additions','all_predictions'):
                assert canonical(formal[key])==canonical(experiment[key]),(mode,stem,key)
            assert len(formal['primary'])<=5 and len(formal['all_predictions'])<=10
            assert formal['all_predictions'][:len(experiment['all_predictions'])-len(experiment['student_additions'])]==formal['primary']+formal['supplementary']+formal['zoom']
            records.append(dict(mode=mode,image=entry['image'],additions=len(formal['student_additions'])))
    assert len(records)==110
    result=dict(status='complete',cases=records,total=110,frozen_selection_parity=True,model_and_manifest_pins_verified=True,new_inference=False)
    (out/'selection_parity.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],total=110,additions=sum(p['additions'] for p in records))))
if __name__=='__main__':main()
