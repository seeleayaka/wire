"""Broader training-only non-disconnection reliability audit; fixed current branch."""
import json,os,shutil,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];REPO=Path('E:/PythonProject10');OUT=ROOT/'artifacts/port_extended_training_controls_20261003'
sys.path.insert(0,str(REPO))
os.environ.update(HF_HUB_OFFLINE='1',YOLO_OFFLINE='True',YOLO_AUTOINSTALL='False',YOLO_CONFIG_DIR=str(OUT/'config'))
from inspection_agent.optional_port_crop_review import sha,read_image,predict
from inspection_agent.context_port_recheck import predict_seed_views,recheck_proposals
from inspection_agent.teacher_student_port_support import native_selection,TEACHER_RELATIVE,TEACHER_SHA,STUDENT_RELATIVE,STUDENT_SHA
from teacher_student_port_policy import merge
from audit_port_multiscale_acceptance import metric
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8');temp.replace(p)
def main():
    if OUT.exists():raise FileExistsError('Fresh output required')
    group=load(ROOT/'artifacts/port_training_multiscale_20261002/protocol.json')
    used={p['image'] for p in load(ROOT/'artifacts/teacher_student_port_20261003/train/report.json')['cases']}
    names=sorted(p for p in group['train_sources'] if p not in used and p.split('_')[0] in ('normal','damaged','misrouted'))
    assert set(names).isdisjoint(group['inner_val_sources']) and names
    (OUT/'config/Ultralytics').mkdir(parents=True);shutil.copy2('C:/Windows/Fonts/arial.ttf',OUT/'config/Ultralytics/Arial.ttf')
    pins={str(p):sha(p) for p in (Path(__file__),Path(__file__).with_name('teacher_student_port_policy.py'),REPO/TEACHER_RELATIVE,REPO/STUDENT_RELATIVE,ROOT/'artifacts/port_training_multiscale_20261002/protocol.json')}
    assert pins[str(REPO/TEACHER_RELATIVE)]==TEACHER_SHA and pins[str(REPO/STUDENT_RELATIVE)]==STUDENT_SHA
    save(OUT/'protocol.json',dict(selection='ALL previously unused training-source normal/damaged/misrouted groups; no label-based picking',
        images=names,source_split='train01',source_group_disjoint=True,unchanged_frozen_current_selection=True,
        teacher_preserved=True,max_primary=5,max_supplementary=5,pins=pins,independent_accuracy=False,source_only=True,reference_not_evaluated=True))
    save(OUT/'progress.json',dict(status='waiting_for_own_training',pid=os.getpid(),total=len(names)))
    # Queue behind the active local training to avoid competing for the same CPU.
    full=ROOT/'artifacts/port_feature_adaptation_20261003/full'
    import psutil
    while not (full/'report.json').exists():
        progress=load(full/'progress.json')
        if progress['status']=='failed' or not psutil.pid_exists(progress['pid']):break
        time.sleep(5)
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(4)
    teachers=YOLO(str(REPO/TEACHER_RELATIVE));students=YOLO(str(REPO/STUDENT_RELATIVE))
    class Capped:
        def __init__(self,model):self.model=model
        def predict(self,*a,**kw):
            result=self.model.predict(*a,**kw);torch.set_num_threads(4);return result
    models=[('teacher',Capped(teachers),TEACHER_SHA),('student',Capped(students),STUDENT_SHA)]
    DATA=REPO/'data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults';cases=[];started=time.monotonic()
    for index,name in enumerate(names,1):
        source=DATA/'images/train01'/name;image=read_image(source);source_sha=sha(source);versions={}
        for key,model,weight_sha in models:
            raw=predict(model,image);case=dict(image=name,source_sha256=source_sha,weight_sha256=weight_sha,predictions=raw,zoom_evidence=[])
            if len(native_selection(case)['supplementary'])<5:case['zoom_evidence']=predict_seed_views(model,image,recheck_proposals(raw))
            versions[key]=case
        fused=merge(versions['teacher'],versions['student']);assert fused['fallback_reason'] is None
        assert sha(source)==source_sha and len(fused['all_predictions'])<=len(fused['primary'])+5
        record=dict(image=name,versions=versions,selection=fused)
        save(OUT/(Path(name).stem+'_predictions.json'),record);cases.append(record)
        save(OUT/'progress.json',dict(status='running',pid=os.getpid(),completed=index,total=len(names),seconds=round(time.monotonic()-started,2)))
        print(f'{index}/{len(names)} {name}: {len(fused["all_predictions"])} cues',flush=True)
    totals={s:{k:0 for k in ('tp','unmatched','fn','predictions','targets')} for s in ('teacher','current')};groups={};rows=[]
    # Labels are used only for final scoring, after all inference and selection.
    for record in cases:
        name=record['image'];h,w=record['versions']['teacher']['predictions']['source_shape'];targets=[];label=DATA/'labels/train01'/(Path(name).stem+'.txt')
        for line in label.read_text(encoding='utf-8').splitlines():
            cls,cx,cy,bw,bh=map(float,line.split())
            if cls in (3,4):targets.append(dict(class_id=int(cls)-3,box=[(cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h]))
        metrics=dict(teacher=metric(native_selection(record['versions']['teacher'])['all_predictions'],targets),current=metric(record['selection']['all_predictions'],targets))
        kind=name.split('_')[0];group=groups.setdefault(kind,{s:{k:0 for k in totals[s]} for s in totals})
        for stage,values in metrics.items():
            for k,v in values.items():totals[stage][k]+=v;group[stage][k]+=v
        rows.append(dict(image=name,metrics=metrics,student_additions=len(record['selection']['student_additions']),label_sha256=sha(label)))
    assert {p:sha(Path(p)) for p in pins}==pins
    report=dict(status='complete',summary=totals,groups=groups,cases=rows,seconds=round(time.monotonic()-started,2),
        training_only=True,source_only=True,reference_gates_not_evaluated=True,production_changed=False,
        purpose='Measure broader training control burden, not a new field accuracy claim',student_added_unmatched=totals['current']['unmatched']-totals['teacher']['unmatched'])
    save(OUT/'report.json',report);save(OUT/'progress.json',dict(status='complete',completed=len(names),total=len(names)));print(json.dumps(report['summary']),flush=True)
if __name__=='__main__':main()
