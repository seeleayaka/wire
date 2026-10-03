"""All246 source selections match the isolated experimental policy."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];REPO=Path('E:/PythonProject10')
sys.path.insert(0,str(REPO));sys.dont_write_bytecode=True
from inspection_agent.feature_residual_port_support import residual_candidates
from inspection_agent.optional_port_crop_review import sha
TRIAL=ROOT/'artifacts/port_residual_feature_support_20261003'
OUT=ROOT/'artifacts/port_residual_feature_formal_20261003_v2'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def canon(rows):return [(p['class_id'],p['box_xyxy'],p['confidence']) for p in rows]
def main():
    if (OUT/'parity.json').exists():raise FileExistsError('Preserve existing proof')
    OUT.mkdir(exist_ok=True);rows=[]
    for mode in ('train','extended','inner','outer'):
        assert load(TRIAL/mode/'report.json')['qualifies']
        for entry in load(TRIAL/mode/'report.json')['cases']:
            stem=Path(entry['image']).stem;record=load(TRIAL/mode/(stem+'_predictions.json'))
            if mode=='extended':
                old=load(ROOT/'artifacts/port_extended_training_controls_20261003'/(stem+'_predictions.json'))
                teacher,student=old['versions']['teacher'],old['versions']['student']
            else:
                teacher=load(ROOT/'artifacts/core_port_recheck_20261002'/mode/(stem+'_zoom_predictions.json'))
                student=load(ROOT/'artifacts/port_training_multiscale_20261002/evaluation'/mode/(stem+'_zoom_predictions.json'))
            formal=residual_candidates(teacher,student,record['feature']);experimental=record['trial']
            assert formal['feature_fallback_reason'] is None
            for key in ('primary','supplementary','zoom','student_additions','feature_additions','all_predictions'):
                assert canon(formal[key])==canon(experimental[key]),(mode,stem,key)
            assert len(formal['all_predictions'])<=len(formal['primary'])+5
            rows.append(dict(mode=mode,image=entry['image'],feature_additions=len(formal['feature_additions'])))
    assert len(rows)==246
    result=dict(status='complete',total=len(rows),canonical_selection_equal=True,cases=rows,
        feature_code_sha256=sha(REPO/'inspection_agent/feature_residual_port_support.py'),
        prediction_replay_only=True,field_accuracy=False)
    (OUT/'parity.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='complete',total=len(rows))))
if __name__=='__main__':main()
