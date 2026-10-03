"""Formal selector must reproduce all270 accepted experiment source selections."""
import json,sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];REPO=Path('E:/PythonProject10');sys.path.insert(0,str(REPO))
from inspection_agent.resolution_loose_plug_support import resolution_candidates,resolution_runtime_fingerprint
from inspection_agent.optional_port_crop_review import sha
BASE=ROOT/'artifacts/resolution_plug_final_budget_20261003';EXTRA=ROOT/'artifacts/remaining_training_ports_20261003'
OUT=ROOT/'artifacts/formal_resolution_parity_20261003'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    if OUT.exists():raise FileExistsError('Preserve evidence')
    frozen=resolution_runtime_fingerprint(REPO);pins={};verified=0
    for stage in ('train','inner','outer'):
        for row in load(BASE/stage/'report.json')['cases']:
            name=row['image'];stem=Path(name).stem;recordpath=BASE/stage/(stem+'_predictions.json');record=load(recordpath)
            if stage=='train' and (EXTRA/(stem+'_predictions.json')).exists():
                teacherfile=EXTRA/(stem+'_predictions.json');teacher=load(teacherfile)['teacher']
            elif stage=='train' and not name.startswith('disconnected_') and (
                ROOT/'artifacts/port_extended_training_controls_20261003'/(stem+'_predictions.json')).exists():
                teacherfile=ROOT/'artifacts/port_extended_training_controls_20261003'/(stem+'_predictions.json');teacher=load(teacherfile)['versions']['teacher']
            else:
                teacherfile=ROOT/'artifacts/core_port_recheck_20261002'/stage/(stem+'_zoom_predictions.json');teacher=load(teacherfile)
            for p in (recordpath,teacherfile):pins[str(p)]=sha(p)
            assert resolution_candidates(teacher,record['current'],record['alternative'])==record['trial'],(stage,name)
            verified+=1
    assert verified==270 and {p:sha(Path(p)) for p in pins}==pins and resolution_runtime_fingerprint(REPO)==frozen
    OUT.mkdir();(OUT/'report.json').write_text(json.dumps(dict(status='complete',verified=verified,exact_including_metadata=True,
        pins=pins,runtime_fingerprint=frozen,new_model_inference=False,field_accuracy=False),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(verified=verified,exact_including_metadata=True)),flush=True)
if __name__=='__main__':main()
