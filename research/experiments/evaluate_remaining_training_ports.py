"""Untuned source-only audit of every remaining training positive scene."""
import json,os,shutil,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];REPO=Path('E:/PythonProject10')
OUT=ROOT/'artifacts/remaining_training_ports_20261003'
DATA=REPO/'data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults'
sys.path.insert(0,str(REPO))
from inspection_agent.optional_port_crop_review import sha,read_image,predict
from inspection_agent.context_port_recheck import predict_seed_views,recheck_proposals
from inspection_agent.teacher_student_port_support import native_selection,TEACHER_RELATIVE,STUDENT_RELATIVE,TEACHER_SHA,STUDENT_SHA
from inspection_agent.feature_residual_port_support import residual_candidates,residual_runtime_fingerprint,WEIGHT_RELATIVE,WEIGHT_SHA
from port_resolution_support import ResolutionModel,append_resolution
from audit_port_multiscale_acceptance import metric,matches
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8');tmp.replace(p)
def main():
    if OUT.exists():raise FileExistsError('Never overwrite an audit')
    groupfile=ROOT/'artifacts/port_training_multiscale_20261002/protocol.json';groups=load(groupfile)
    base=ROOT/'artifacts/port_residual_feature_support_20261003'
    seen={r['image'] for mode in ('train','extended') for r in load(base/mode/'report.json')['cases']}
    names=sorted(set(groups['train_sources'])-seen)
    assert len(names)==24 and set(names).isdisjoint(groups['inner_val_sources'])
    assert all(n.startswith('disconnected_') for n in names)
    paths=[REPO/relative for relative in (TEACHER_RELATIVE,STUDENT_RELATIVE,WEIGHT_RELATIVE)]
    expected=(TEACHER_SHA,STUDENT_SHA,WEIGHT_SHA)
    assert tuple(sha(p) for p in paths)==expected
    runtime=residual_runtime_fingerprint(REPO)
    pins={str(p):sha(p) for p in [Path(__file__),Path(__file__).with_name('port_resolution_support.py'),
        Path(__file__).with_name('audit_port_multiscale_acceptance.py'),groupfile,*paths]}
    for name in names:pins[str(DATA/'images/train01'/name)]=sha(DATA/'images/train01'/name)
    (OUT/'config/Ultralytics').mkdir(parents=True);shutil.copy2('C:/Windows/Fonts/arial.ttf',OUT/'config/Ultralytics/Arial.ttf')
    save(OUT/'protocol.json',dict(images=names,pins=pins,runtime_fingerprint=runtime,
        source_selection='ALL192 training sources minus32 original controls minus136 extended negatives',
        remaining24_policy_unaudited_but_model_training_seen=True,
        no_validation_images=True,no_threshold_or_budget_change=True,predictions_before_labels=True,
        baseline='accepted teacher+student+feature',trial='append1280 same feature checkpoint',
        source_only=True,reference_pending=True,production_changed=False,field_accuracy=False))
    os.environ.update(YOLO_CONFIG_DIR=str(OUT/'config'),YOLO_OFFLINE='True',YOLO_AUTOINSTALL='False',HF_HUB_OFFLINE='1')
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(4);models=[YOLO(str(p)) for p in paths]
    assert all(m.task=='segment' and dict(m.names)=={0:'unplugged_plug',1:'unplugged_jack'} for m in models)
    wrapped=[ResolutionModel(m,960,lambda:torch.set_num_threads(4)) for m in models]
    high=ResolutionModel(models[-1],1280,lambda:torch.set_num_threads(4))
    started=time.monotonic();records=[]
    try:
        for index,name in enumerate(names):
            source=DATA/'images/train01'/name;image=read_image(source)
            save(OUT/'progress.json',dict(status='running',pid=os.getpid(),completed=index,total=len(names),image=name,phase='inference'))
            def infer(model,digest,resolution):
                raw=predict(model,image)
                result=dict(image=name,source_sha256=pins[str(source)],weight_sha256=digest,
                    inference_imgsz=resolution,predictions=raw,zoom_evidence=[])
                if len(native_selection(result)['supplementary'])<5:
                    result['zoom_evidence']=predict_seed_views(model,image,recheck_proposals(raw))
                return result
            teacher,student,feature=[infer(m,digest,960) for m,digest in zip(wrapped,expected)]
            current=residual_candidates(teacher,student,feature)
            assert not current['feature_fallback_reason']
            room=len(current['all_predictions'])-len(current['primary'])<5
            support=any(p['confidence']>.25 for p in teacher['predictions']['merged_predictions'])
            if room and support:alternative=infer(high,WEIGHT_SHA,1280)
            else:alternative=dict(image=name,source_sha256=pins[str(source)],weight_sha256=WEIGHT_SHA,inference_imgsz=1280,
                predictions=dict(source_shape=list(image.shape[:2]),merged_predictions=[],edge_kept_predictions=[]),zoom_evidence=[])
            trial=append_resolution(teacher,current,alternative);assert not trial['resolution_fallback_reason']
            save(OUT/(Path(name).stem+'_predictions.json'),dict(image=name,teacher=teacher,student=student,feature=feature,
                alternative=alternative,current=current,trial=trial))
            h,w=image.shape[:2];label=DATA/'labels/train01'/(Path(name).stem+'.txt');targets=[]
            for line in label.read_text(encoding='utf-8').splitlines():
                cls,cx,cy,bw,bh=map(float,line.split())
                if cls in (3,4):targets.append(dict(class_id=int(cls)-3,box=[(cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h]))
            old,new=current['all_predictions'],trial['all_predictions']
            oldhits,newhits=matches(old,targets)[0],matches(new,targets)[0]
            row=dict(image=name,metrics=dict(current=metric(old,targets),trial=metric(new,targets)),
                gained=sorted(newhits-oldhits),lost=sorted(oldhits-newhits),additions=len(trial['resolution_additions']),label_sha256=sha(label))
            records.append(row);print(json.dumps(row),flush=True)
            save(OUT/'partial.json',dict(cases=records,seconds=round(time.monotonic()-started,2)))
        assert {p:sha(Path(p)) for p in pins}==pins and residual_runtime_fingerprint(REPO)==runtime
        totals={s:{key:sum(r['metrics'][s][key] for r in records) for key in ('tp','unmatched','fn','predictions','targets')} for s in ('current','trial')}
        qualifies=totals['trial']['tp']>=totals['current']['tp'] and totals['trial']['unmatched']<=totals['current']['unmatched'] and not any(r['lost'] for r in records)
        report=dict(status='complete',qualifies=qualifies,summary=totals,cases=records,seconds=round(time.monotonic()-started,2),
            source_only=True,reference_pending=True,no_automatic_deployment=True,field_accuracy=False)
        save(OUT/'report.json',report);save(OUT/'progress.json',dict(status='complete',qualifies=qualifies,summary=totals))
    except BaseException as error:
        save(OUT/'progress.json',dict(status='failed',error=type(error).__name__+': '+str(error)));raise
if __name__=='__main__':main()
