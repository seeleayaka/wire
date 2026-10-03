"""Frozen training-first student-seeded teacher context study. No current-mainline edits."""
import argparse,json,os,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];REPO=Path('E:/PythonProject10');OUT=ROOT/'artifacts/teacher_context_student_20261003'
sys.path.insert(0,str(REPO))
os.environ.update(HF_HUB_OFFLINE='1',YOLO_OFFLINE='True',YOLO_AUTOINSTALL='False',YOLO_CONFIG_DIR=str(OUT/'config'))
from inspection_agent.optional_port_crop_review import sha,read_image
from inspection_agent.context_port_recheck import predict_seed_views
from inspection_agent.teacher_student_port_support import TEACHER_SHA,TEACHER_RELATIVE,STUDENT_SHA
from teacher_context_student_policy import proposals,extend,POLICY
from teacher_student_port_policy import merge
from audit_port_multiscale_acceptance import metric
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['train','inner','outer'],required=True);args=parser.parse_args();mode=args.mode
    proto=load(OUT/'protocol.json');output=OUT/mode
    if output.exists():raise FileExistsError('Keep earlier runs')
    if mode!='train':
        decision=load(OUT/'training_decision.json');assert decision['accepted']
        assert decision['policy_sha256']==sha(Path(__file__).with_name('teacher_context_student_policy.py'))
        assert decision['evaluator_sha256']==sha(Path(__file__)) and decision['protocol_sha256']==sha(OUT/'protocol.json')
    output.mkdir();(OUT/'config/Ultralytics').mkdir(parents=True,exist_ok=True)
    import shutil;shutil.copy2('C:/Windows/Fonts/arial.ttf',OUT/'config/Ultralytics/Arial.ttf')
    import torch;torch.set_num_threads(4)
    from ultralytics import YOLO
    assert sha(REPO/TEACHER_RELATIVE)==TEACHER_SHA
    model=YOLO(str(REPO/TEACHER_RELATIVE));assert model.task=='segment'
    DATA=REPO/'data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults'
    prior=load(ROOT/'artifacts/teacher_student_port_20261003'/mode/'report.json');cases=[];pins={};started=time.monotonic();split='val01' if mode=='outer' else 'train01'
    # Complete every prediction before reading any source labels.
    for item in prior['cases']:
        name=item['image'];stem=Path(name).stem;source=DATA/'images'/split/name
        tp=ROOT/'artifacts/core_port_recheck_20261002'/mode/(stem+'_zoom_predictions.json')
        sp=ROOT/'artifacts/port_training_multiscale_20261002/evaluation'/mode/(stem+'_zoom_predictions.json')
        teacher=load(tp);student=load(sp)
        assert teacher['source_sha256']==student['source_sha256']==sha(source)
        assert teacher['weight_sha256']==TEACHER_SHA and student['weight_sha256']==STUDENT_SHA
        for p in (source,tp,sp):pins[str(p)]=sha(p)
        seeds=proposals(teacher,student);entries=predict_seed_views(model,read_image(source),seeds) if seeds else []
        selected=extend(teacher,student,entries);base=merge(teacher,student)
        assert selected['all_predictions'][:len(base['all_predictions'])]==base['all_predictions']
        assert len(selected['all_predictions'])<=len(base['primary'])+5
        record=dict(image=name,source_shape=teacher['predictions']['source_shape'],teacher_context=entries,selected=selected,baseline=base,proposals=len(seeds))
        save(output/(stem+'_predictions.json'),record);cases.append(record)
        print(json.dumps(dict(image=name,seeds=len(seeds),added=len(selected['teacher_context_additions']))),flush=True)
    total={s:{k:0 for k in ('tp','unmatched','fn','predictions','targets')} for s in ('baseline','context')};normal={s:0 for s in total};rows=[]
    for case in cases:
        name=case['image'];h,w=case['source_shape'];targets=[];label=DATA/'labels'/split/(Path(name).stem+'.txt')
        for line in label.read_text(encoding='utf-8').splitlines():
            cls,cx,cy,bw,bh=map(float,line.split())
            if cls in (3,4):targets.append(dict(class_id=int(cls)-3,box=[(cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h]))
        metrics={s:metric(case[key]['all_predictions'],targets) for s,key in (('baseline','baseline'),('context','selected'))}
        for s,values in metrics.items():
            for k,v in values.items():total[s][k]+=v
            if name.startswith('normal_'):normal[s]+=values['predictions']
        rows.append(dict(image=name,metrics=metrics,seeds=case['proposals'],additions=len(case['selected']['teacher_context_additions'])))
    assert total['baseline']==prior['summary']['cross_model_supported']
    assert {p:sha(Path(p)) for p in pins}==pins and sha(REPO/TEACHER_RELATIVE)==TEACHER_SHA
    accepted=(total['context']['tp']>total['baseline']['tp'] if mode!='outer' else total['context']['tp']>=total['baseline']['tp']) and total['context']['unmatched']<=total['baseline']['unmatched'] and normal['context']<=normal['baseline']
    report=dict(status='complete',summary=total,normal_cues=normal,qualifies=accepted,cases=rows,seconds=round(time.monotonic()-started,2),
        cached_full_predictions=True,fresh_teacher_context=True,new_sam=False,old_cues_preserved=True,field_accuracy=False,thresholds_frozen=True)
    save(output/'inputs.json',pins);save(output/'report.json',report)
    if mode=='train':save(OUT/'training_decision.json',dict(accepted=accepted,policy_sha256=sha(Path(__file__).with_name('teacher_context_student_policy.py')),
        evaluator_sha256=sha(Path(__file__)),protocol_sha256=sha(OUT/'protocol.json'),selection_on_training_only=True))
    print(json.dumps(dict(summary=total,qualifies=accepted,seconds=report['seconds'])),flush=True)
if __name__=='__main__':main()
