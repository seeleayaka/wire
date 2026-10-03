"""One frozen change:1280 rather than960 inference, training and burden gates first."""
import json,os,shutil,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];REPO=Path('E:/PythonProject10');sys.path.insert(0,str(REPO))
BASE=ROOT/'artifacts/port_residual_feature_support_20261003'
OUT=ROOT/'artifacts/port_high_resolution_support_20261003'
DATA=REPO/'data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults'
from inspection_agent.optional_port_crop_review import sha,read_image,predict
from inspection_agent.context_port_recheck import predict_seed_views,recheck_proposals
from inspection_agent.teacher_student_port_support import native_selection,TEACHER_SHA
from inspection_agent.feature_residual_port_support import WEIGHT_SHA,WEIGHT_RELATIVE,residual_runtime_fingerprint
from port_resolution_support import append_resolution,ResolutionModel
from audit_port_multiscale_acceptance import metric,matches
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8');temp.replace(p)
def qualifies(mode,old,new,normal,losses):
    gain=new['tp']>old['tp'] if mode in ('train','inner') else new['tp']>=old['tp']
    return gain and new['unmatched']<=old['unmatched'] and normal==0 and losses==0
def main():
    if OUT.exists():raise FileExistsError('Never duplicate pinned experiment')
    weight=REPO/WEIGHT_RELATIVE;assert sha(weight)==WEIGHT_SHA
    pins={str(p):sha(p) for p in (Path(__file__),Path(__file__).with_name('port_resolution_support.py'),
        Path(__file__).with_name('teacher_student_port_policy.py'),Path(__file__).with_name('core_port_recheck_policy.py'),
        Path(__file__).with_name('core_port_supplement_policy.py'),Path(__file__).with_name('core_port_precision_policy.py'),
        Path(__file__).with_name('audit_port_multiscale_acceptance.py'),weight,
        REPO/'inspection_agent/optional_port_crop_review.py',REPO/'inspection_agent/context_port_recheck.py')}
    modes=('train','extended','inner','outer');inputs={};before=residual_runtime_fingerprint(REPO)
    for mode in modes:
        report=load(BASE/mode/'report.json');assert report['qualifies'];rows=[]
        for case in report['cases']:
            name=case['image'];stem=Path(name).stem;baseline_file=BASE/mode/(stem+'_predictions.json')
            baseline=load(baseline_file);current=baseline['trial']
            if mode=='extended':
                teacher_file=ROOT/'artifacts/port_extended_training_controls_20261003'/(stem+'_predictions.json')
                teacher=load(teacher_file)['versions']['teacher']
            else:
                teacher_file=ROOT/'artifacts/core_port_recheck_20261002'/mode/(stem+'_zoom_predictions.json')
                teacher=load(teacher_file)
            source=DATA/'images'/('val01' if mode=='outer' else 'train01')/name
            for p in (baseline_file,teacher_file,source):pins[str(p)]=sha(p)
            assert teacher['source_sha256']==pins[str(source)] and teacher['weight_sha256']==TEACHER_SHA
            rows.append(dict(image=name,source=source,teacher=teacher,current=current))
        assert len(rows)==dict(train=32,extended=136,inner=48,outer=30)[mode];inputs[mode]=rows
    (OUT/'config/Ultralytics').mkdir(parents=True);shutil.copy2('C:/Windows/Fonts/arial.ttf',OUT/'config/Ultralytics/Arial.ttf')
    save(OUT/'protocol.json',dict(pins=pins,input_resolution=1280,original_resolution=960,same_final_checkpoint=WEIGHT_SHA,
        unchanged_windows_and_nms=True,unchanged_thresholds=True,source_prediction_selection_before_labels=True,
        training_then_all136_burden_then_inner_then_outer=True,preserve_accepted_three_model_cues=True,max_primary=5,max_extra=5,
        validation_already_seen=True,field_accuracy=False,no_automatic_deployment=True,runtime_fingerprint=before))
    os.environ.update(YOLO_CONFIG_DIR=str(OUT/'config'),YOLO_OFFLINE='True',YOLO_AUTOINSTALL='False',HF_HUB_OFFLINE='1')
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(4);model=YOLO(str(weight));assert model.task=='segment' and dict(model.names)=={0:'unplugged_plug',1:'unplugged_jack'}
    wrapped=ResolutionModel(model,1280,lambda:torch.set_num_threads(4));started=time.monotonic()
    try:
        for mode in modes:
            folder=OUT/mode;folder.mkdir();records=[];inferred=0;skipped=0
            for index,case in enumerate(inputs[mode],1):
                teacher,current=case['teacher'],case['current']
                supported=any(p['confidence']>.25 for p in teacher['predictions']['merged_predictions'])
                room=len(current['all_predictions'])-len(current['primary'])<5
                if supported and room:
                    image=read_image(case['source']);raw=predict(wrapped,image)
                    alternative=dict(image=case['image'],source_sha256=teacher['source_sha256'],weight_sha256=WEIGHT_SHA,
                        inference_imgsz=1280,predictions=raw,zoom_evidence=[])
                    if len(native_selection(alternative)['supplementary'])<5:
                        alternative['zoom_evidence']=predict_seed_views(wrapped,image,recheck_proposals(raw));torch.set_num_threads(4)
                    inferred+=1
                else:
                    alternative=dict(image=case['image'],source_sha256=teacher['source_sha256'],weight_sha256=WEIGHT_SHA,inference_imgsz=1280,
                        predictions=dict(source_shape=teacher['predictions']['source_shape'],merged_predictions=[],edge_kept_predictions=[]),zoom_evidence=[])
                    skipped+=1
                result=append_resolution(teacher,current,alternative);assert result['resolution_fallback_reason'] is None
                record=dict(image=case['image'],current=current,trial=result,alternative=alternative)
                save(folder/(Path(case['image']).stem+'_predictions.json'),record);records.append(record)
                save(OUT/'progress.json',dict(status='running',pid=os.getpid(),mode=mode,completed=index,total=len(inputs[mode]),
                    inferred=inferred,exact_short_circuits=skipped,seconds=round(time.monotonic()-started,2)))
                print(f'{mode} {index}/{len(inputs[mode])} {case["image"]} added={len(result["resolution_additions"])}',flush=True)
            totals={s:dict(tp=0,unmatched=0,fn=0,predictions=0,targets=0) for s in ('current','trial')};normal=0;lost=0;cases=[]
            for record in records:
                name=record['image'];label=DATA/'labels'/('val01' if mode=='outer' else 'train01')/(Path(name).stem+'.txt')
                h,w=record['alternative']['predictions']['source_shape'];targets=[]
                for line in label.read_text(encoding='utf-8').splitlines():
                    cls,cx,cy,bw,bh=map(float,line.split())
                    if cls in (3,4):targets.append(dict(class_id=int(cls)-3,box=[(cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h]))
                old,new=record['current']['all_predictions'],record['trial']['all_predictions']
                values=dict(current=metric(old,targets),trial=metric(new,targets))
                old_hits,new_hits=matches(old,targets)[0],matches(new,targets)[0];lost+=len(old_hits-new_hits)
                if name.startswith('normal_'):normal+=len(new)
                for stage,metrics in values.items():
                    for k,v in metrics.items():totals[stage][k]+=v
                cases.append(dict(image=name,metrics=values,additions=len(record['trial']['resolution_additions']),
                    gained=sorted(new_hits-old_hits),lost=sorted(old_hits-new_hits),label_sha256=sha(label)))
            assert totals['current']==load(BASE/mode/'report.json')['summary']['trial']
            assert {p:sha(Path(p)) for p in pins}==pins and residual_runtime_fingerprint(REPO)==before
            passed=qualifies(mode,totals['current'],totals['trial'],normal,lost)
            report=dict(status='complete',qualifies=passed,summary=totals,normal_cues=normal,lost=lost,cases=cases,
                inferred=inferred,exact_short_circuits=skipped,seconds=round(time.monotonic()-started,2),
                reference_not_evaluated=True,source_only=True,field_accuracy=False,production_changed=False)
            save(folder/'report.json',report);print(json.dumps(dict(mode=mode,summary=totals,qualifies=passed,lost=lost)),flush=True)
            if not passed:
                save(OUT/'progress.json',dict(status='rejected',mode=mode,summary=totals,no_automatic_deployment=True));return
        save(OUT/'progress.json',dict(status='source_only_pass_requires_live_reference',no_automatic_deployment=True,seconds=round(time.monotonic()-started,2)))
    except BaseException as error:
        save(OUT/'progress.json',dict(status='failed',error=type(error).__name__+': '+str(error)));raise
if __name__=='__main__':main()
