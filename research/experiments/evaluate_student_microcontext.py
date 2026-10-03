"""Training-first novel student microcontext study, cached full images/fresh crops."""
import argparse,json,os,sys,time,shutil
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];REPO=Path('E:/PythonProject10');OUT=ROOT/'artifacts/student_microcontext_20261003'
sys.path.insert(0,str(REPO))
os.environ.update(HF_HUB_OFFLINE='1',YOLO_OFFLINE='True',YOLO_AUTOINSTALL='False',YOLO_CONFIG_DIR=str(OUT/'config'))
from inspection_agent.optional_port_crop_review import sha,read_image
from inspection_agent.teacher_student_port_support import TEACHER_SHA,TEACHER_RELATIVE,STUDENT_SHA,STUDENT_RELATIVE
from student_microcontext_policy import proposals,extend,predict_views,POLICY
from teacher_student_port_policy import merge
from audit_port_multiscale_acceptance import metric
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['train','inner','outer'],required=True);args=parser.parse_args();mode=args.mode
    proto=load(OUT/'protocol.json');output=OUT/mode
    if output.exists():raise FileExistsError('Keep earlier runs')
    hashes=dict(policy_sha256=sha(Path(__file__).with_name('student_microcontext_policy.py')),
        confirmation_sha256=sha(Path(__file__).with_name('teacher_context_student_policy.py')),
        parent_policy_sha256=sha(Path(__file__).with_name('teacher_student_port_policy.py')),
        evaluator_sha256=sha(Path(__file__)),protocol_sha256=sha(OUT/'protocol.json'))
    if mode!='train':
        decision=load(OUT/'training_decision.json');assert decision['accepted']
        for k,v in hashes.items():assert decision[k]==v,(k,'changed')
    output.mkdir();(OUT/'config/Ultralytics').mkdir(parents=True,exist_ok=True)
    shutil.copy2('C:/Windows/Fonts/arial.ttf',OUT/'config/Ultralytics/Arial.ttf')
    import torch;torch.set_num_threads(4)
    from ultralytics import YOLO
    assert sha(REPO/TEACHER_RELATIVE)==TEACHER_SHA and sha(REPO/STUDENT_RELATIVE)==STUDENT_SHA
    model=YOLO(str(REPO/STUDENT_RELATIVE));assert model.task=='segment' and dict(model.names)=={0:'unplugged_plug',1:'unplugged_jack'}
    DATA=REPO/'data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults'
    prior=load(ROOT/'artifacts/teacher_student_port_20261003'/mode/'report.json');cases=[];pins={};started=time.monotonic();split='val01' if mode=='outer' else 'train01'
    train_split=load(ROOT/'artifacts/port_training_multiscale_20261002/protocol.json')
    for item in prior['cases']:
        name=item['image'];stem=Path(name).stem;source=DATA/'images'/split/name
        if mode=='train':assert name in train_split['train_sources']
        if mode=='inner':assert name in train_split['inner_val_sources'] and name not in train_split['train_sources']
        tp=ROOT/'artifacts/core_port_recheck_20261002'/mode/(stem+'_zoom_predictions.json')
        sp=ROOT/'artifacts/port_training_multiscale_20261002/evaluation'/mode/(stem+'_zoom_predictions.json')
        teacher=load(tp);student=load(sp)
        assert teacher['source_sha256']==student['source_sha256']==sha(source)
        assert teacher['weight_sha256']==TEACHER_SHA and student['weight_sha256']==STUDENT_SHA
        for p in (source,tp,sp):pins[str(p)]=sha(p)
        seeds=proposals(teacher,student);entries=predict_views(model,read_image(source),seeds) if seeds else []
        selected=extend(teacher,student,entries);base=merge(teacher,student)
        assert selected['all_predictions'][:len(base['all_predictions'])]==base['all_predictions']
        assert len(selected['all_predictions'])<=len(base['primary'])+5
        record=dict(image=name,source_shape=teacher['predictions']['source_shape'],microcontext=entries,selected=selected,baseline=base,proposals=len(seeds))
        save(output/(stem+'_predictions.json'),record);cases.append(record)
        print(json.dumps(dict(image=name,seeds=len(seeds),added=len(selected['microcontext_additions']))),flush=True)
    total={s:{k:0 for k in ('tp','unmatched','fn','predictions','targets')} for s in ('baseline','microcontext')};normal={s:0 for s in total};rows=[]
    for case in cases:
        name=case['image'];h,w=case['source_shape'];targets=[];label=DATA/'labels'/split/(Path(name).stem+'.txt')
        for line in label.read_text(encoding='utf-8').splitlines():
            cls,cx,cy,bw,bh=map(float,line.split())
            if cls in (3,4):targets.append(dict(class_id=int(cls)-3,box=[(cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h]))
        metrics={s:metric(case[key]['all_predictions'],targets) for s,key in (('baseline','baseline'),('microcontext','selected'))}
        for s,values in metrics.items():
            for k,v in values.items():total[s][k]+=v
            if name.startswith('normal_'):normal[s]+=values['predictions']
        rows.append(dict(image=name,metrics=metrics,seeds=case['proposals'],additions=len(case['selected']['microcontext_additions'])))
        save(output/(Path(name).stem+'_evaluation.json'),dict(image=name,metrics=metrics,label_sha256=sha(label)))
    assert total['baseline']==prior['summary']['cross_model_supported']
    assert {p:sha(Path(p)) for p in pins}==pins and sha(REPO/STUDENT_RELATIVE)==STUDENT_SHA
    accepted=(total['microcontext']['tp']>total['baseline']['tp'] if mode!='outer' else total['microcontext']['tp']>=total['baseline']['tp']) and total['microcontext']['unmatched']<=total['baseline']['unmatched'] and normal['microcontext']<=normal['baseline']
    report=dict(status='complete',summary=total,normal_cues=normal,qualifies=accepted,cases=rows,seconds=round(time.monotonic()-started,2),
        cached_full_predictions=True,fresh_student_microcontext=True,new_sam=False,old_cues_preserved=True,field_accuracy=False,thresholds_frozen=True)
    save(output/'inputs.json',pins);save(output/'report.json',report)
    if mode=='train':save(OUT/'training_decision.json',dict(accepted=accepted,**hashes,selection_on_training_only=True))
    print(json.dumps(dict(summary=total,qualifies=accepted,seconds=report['seconds'])),flush=True)
if __name__=='__main__':main()
