"""Fresh fixed-final model inference, teacher-preserved merge, training-first gates."""
import argparse,json,os,shutil,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];REPO=Path('E:/PythonProject10');BASE=ROOT/'artifacts/port_feature_adaptation_20261003'
OLD=ROOT/'artifacts/core_port_recheck_20261002';ACCEPTED=ROOT/'artifacts/teacher_student_port_20261003'
DATA=REPO/'data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults'
sys.path.insert(0,str(REPO))
from inspection_agent.optional_port_crop_review import sha,read_image,predict
from inspection_agent.context_port_recheck import predict_seed_views,recheck_proposals
from inspection_agent.teacher_student_port_support import native_selection,STUDENT_SHA,STUDENT_RELATIVE,TEACHER_SHA,TEACHER_RELATIVE
from teacher_student_port_policy import merge
from audit_port_multiscale_acceptance import metric,matches
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8');temp.replace(p)
def qualifies(mode,current,candidate,normal,losses):
    return ((candidate['tp']>current['tp'] if mode!='outer' else candidate['tp']>=current['tp'])
        and candidate['unmatched']<=current['unmatched'] and normal==0 and losses==0)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['train','inner','outer'],required=True);args=parser.parse_args();mode=args.mode
    protocol=load(BASE/'protocol.json');full=load(BASE/'full/report.json');assert full['status']=='complete' and full['completed_epochs']==2
    assert full['nonzero_gradient_steps']>0 and full['changed_new_backbone_tensors']>0 and full['frozen_parameters_changed']==0
    assert full['initialization_sha256']==STUDENT_SHA and full['protocol_sha256']==sha(BASE/'protocol.json')
    assert {p:sha(Path(p)) for p in full['pins']}==full['pins']
    candidate=Path(full['weights']['last.pt']['path']);weight_sha=sha(candidate);assert weight_sha==full['weights']['last.pt']['sha256'] and weight_sha!=STUDENT_SHA
    pins={str(p):sha(p) for p in (Path(__file__),Path(__file__).with_name('teacher_student_port_policy.py'),BASE/'protocol.json',REPO/TEACHER_RELATIVE,REPO/STUDENT_RELATIVE,REPO/'inspection_agent/teacher_student_port_support.py',REPO/'inspection_agent/context_port_recheck.py',candidate)}
    if mode!='train':
        decision=load(BASE/'evaluation/train/report.json');assert decision['qualifies']
        train_pins=load(BASE/'evaluation/train/protocol.json')['pins']
        for p,value in pins.items():assert train_pins[p]==value
    target=BASE/'evaluation'/mode
    if target.exists():raise FileExistsError('Fresh output required')
    current_report=load(ACCEPTED/mode/'report.json');names=[p['image'] for p in current_report['cases']];assert len(names)=={'train':32,'inner':48,'outer':30}[mode]
    split='val01' if mode=='outer' else 'train01';groups=load(ROOT/'artifacts/port_training_multiscale_20261002/protocol.json')
    assert set(groups['train_sources']).isdisjoint(groups['inner_val_sources'])
    if mode=='train':assert set(names)<=set(groups['train_sources'])
    if mode=='inner':assert set(names)==set(groups['inner_val_sources'])
    cases=[]
    for name in names:
        old_path=OLD/mode/(Path(name).stem+'_zoom_predictions.json');current_path=ACCEPTED/mode/(Path(name).stem+'_evaluation.json')
        teacher=load(old_path);current=load(current_path);source=DATA/'images'/split/name
        assert teacher['weight_sha256']==TEACHER_SHA and teacher['source_sha256']==sha(source)
        for p in (old_path,current_path,source):pins[str(p)]=sha(p)
        cases.append(dict(image=name,teacher=teacher,current=current))
    (target/'config/Ultralytics').mkdir(parents=True);shutil.copy2('C:/Windows/Fonts/arial.ttf',target/'config/Ultralytics/Arial.ttf')
    save(target/'protocol.json',dict(pins=pins,candidate_checkpoint='fixed last.pt',prediction_selection_before_labels=True,
        scope='Same-camera port localization, not field fault or continuity accuracy',outer_training=False))
    os.environ.update(YOLO_CONFIG_DIR=str(target/'config'),YOLO_OFFLINE='True',YOLO_AUTOINSTALL='False',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(4);model=YOLO(str(candidate));assert model.task=='segment' and dict(model.names)=={0:'unplugged_plug',1:'unplugged_jack'}
    class Capped:
        def predict(self,*a,**kw):
            outputs=model.predict(*a,**kw);torch.set_num_threads(4);return outputs
    capped=Capped();started=time.monotonic()
    # Inference and source-only candidate selection for every source before labels.
    for index,case in enumerate(cases,1):
        name=case['image'];image=read_image(DATA/'images'/split/name);tick=time.monotonic();raw=predict(capped,image)
        student=dict(image=name,source_sha256=case['teacher']['source_sha256'],weight_sha256=weight_sha,predictions=raw,zoom_evidence=[])
        if len(native_selection(student)['supplementary'])<5:student['zoom_evidence']=predict_seed_views(capped,image,recheck_proposals(raw))
        selected=merge(case['teacher'],student);assert selected['fallback_reason'] is None
        assert len(selected['primary'])<=5 and len(selected['all_predictions'])<=len(selected['primary'])+5
        case.update(student=student,selected=selected)
        save(target/(Path(name).stem+'_zoom_predictions.json'),student)
        save(target/'progress.json',dict(status='running',completed=index,total=len(cases),seconds=round(time.monotonic()-started,2)))
        print(f'{index}/{len(cases)} {name}: {time.monotonic()-tick:.2f}s',flush=True)
    totals={s:{k:0 for k in ('tp','unmatched','fn','predictions','targets')} for s in ('current','candidate','student_only')};normal={s:0 for s in totals};records=[];lost=0
    for case in cases:
        name=case['image'];label=DATA/'labels'/split/(Path(name).stem+'.txt');assert sha(label)==case['current']['label_sha256']
        h,w=case['student']['predictions']['source_shape'];targets=[]
        for line in label.read_text(encoding='utf-8').splitlines():
            cls,cx,cy,bw,bh=map(float,line.split())
            if cls in (3,4):targets.append(dict(class_id=int(cls)-3,box=[(cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h]))
        selections=dict(current=case['current']['selected']['cross_model_supported']['all_predictions'],candidate=case['selected']['all_predictions'],student_only=native_selection(case['student'])['all_predictions'])
        metrics={s:metric(rows,targets) for s,rows in selections.items()}
        for s,values in metrics.items():
            for k,v in values.items():totals[s][k]+=v
            if name.startswith('normal_'):normal[s]+=values['predictions']
        old_hits=matches(selections['current'],targets)[0];new_hits=matches(selections['candidate'],targets)[0];lost+=len(old_hits-new_hits)
        row=dict(image=name,metrics=metrics,lost_targets=sorted(old_hits-new_hits),gained_targets=sorted(new_hits-old_hits),selected=case['selected'],label_sha256=sha(label))
        save(target/(Path(name).stem+'_evaluation.json'),row);records.append({k:v for k,v in row.items() if k not in ('selected','label_sha256')})
    assert totals['current']==current_report['summary']['cross_model_supported'] and {p:sha(Path(p)) for p in pins}==pins
    accepted=qualifies(mode,totals['current'],totals['candidate'],normal['candidate'],lost)
    report=dict(status='complete',summary=totals,normal_cues=normal,lost_current_targets=lost,qualifies=accepted,cases=records,
        seconds=round(time.monotonic()-started,2),source_only=True,new_full_source_inference=True,new_sam=False,
        production_approved=False,same_camera_previously_seen=True,field_accuracy=False)
    save(target/'report.json',report);save(target/'progress.json',dict(status='complete',completed=len(cases),total=len(cases)))
    print(json.dumps(dict(summary=totals,qualifies=accepted,lost_current_targets=lost)),flush=True)
if __name__=='__main__':main()
