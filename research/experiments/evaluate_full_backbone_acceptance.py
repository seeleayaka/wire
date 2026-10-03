"""Final fixed checkpoint: ALL192 training gate, then inner48 and outer30."""
import copy,json,os,shutil,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];REPO=Path('E:/PythonProject10');sys.path.insert(0,str(REPO))
BASE=ROOT/'artifacts/port_full_backbone_20261003';OUT=BASE/'evaluation'
HIGH=ROOT/'artifacts/port_high_resolution_support_20261003';EXTRA=ROOT/'artifacts/remaining_training_ports_20261003'
DATA=REPO/'data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults'
from inspection_agent.optional_port_crop_review import sha,read_image,predict
from inspection_agent.context_port_recheck import predict_seed_views,recheck_proposals
from inspection_agent.teacher_student_port_support import complementary_candidates,native_selection
from inspection_agent.feature_residual_port_support import residual_runtime_fingerprint
from inspection_agent.port_tiling import box_iou
from audit_port_multiscale_acceptance import metric,matches
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8');tmp.replace(p)
def append_backbone(teacher,current,newcase):
    output=copy.deepcopy(current);output.update(backbone_additions=[],backbone_fallback_reason=None)
    candidates=complementary_candidates(teacher,newcase)
    if candidates['fallback_reason']:output['backbone_fallback_reason']=candidates['fallback_reason'];return output
    remaining=5-(len(current['all_predictions'])-len(current['primary']))
    for row in candidates['student_additions']:
        if len(output['backbone_additions'])>=remaining:break
        if any(box_iou(row['box_xyxy'],old['box_xyxy'])>=.5 for old in output['all_predictions']):continue
        row=copy.deepcopy(row);row.update(evidence_tier='full_backbone_residual_manual_review',inference_imgsz=960)
        output['backbone_additions'].append(row);output['all_predictions'].append(row)
    assert output['all_predictions'][:len(current['all_predictions'])]==current['all_predictions']
    assert len(output['all_predictions'])<=len(output['primary'])+5
    return output
def main():
    if OUT.exists():raise FileExistsError('No repeated final-checkpoint evaluation')
    training=load(BASE/'full/report.json');assert training['status']=='complete' and training['completed_epochs']==2
    assert training['changed_early_backbone_tensors']>0 and training['frozen_parameters_changed']==0
    weight=Path(training['weights']['last.pt']['path']);digest=training['weights']['last.pt']['sha256'];assert sha(weight)==digest
    runtime=residual_runtime_fingerprint(REPO);assert runtime==training['runtime_fingerprint']
    pins={str(p):sha(p) for p in (Path(__file__),BASE/'full/report.json',BASE/'full/protocol.json',weight,
        Path(__file__).with_name('audit_port_multiscale_acceptance.py'),REPO/'inspection_agent/teacher_student_port_support.py',
        REPO/'inspection_agent/context_port_recheck.py',REPO/'inspection_agent/optional_port_crop_review.py')}
    stages=dict(train=[HIGH/'train',HIGH/'extended',EXTRA],inner=[HIGH/'inner'],outer=[HIGH/'outer']);inputs={}
    for stage,folders in stages.items():
        inputs[stage]=[]
        for folder in folders:
            for row in load(folder/'report.json')['cases']:
                name=row['image'];stem=Path(name).stem;path=folder/(stem+'_predictions.json');record=load(path)
                if folder==EXTRA:teacher=record['teacher'];teacherfile=path
                elif folder.name=='extended':
                    teacherfile=ROOT/'artifacts/port_extended_training_controls_20261003'/(stem+'_predictions.json')
                    teacher=load(teacherfile)['versions']['teacher']
                else:
                    teacherfile=ROOT/'artifacts/core_port_recheck_20261002'/stage/(stem+'_zoom_predictions.json');teacher=load(teacherfile)
                source=DATA/'images'/('val01' if stage=='outer' else 'train01')/name
                for p in (path,teacherfile,source):pins[str(p)]=sha(p)
                assert teacher['source_sha256']==pins[str(source)]
                inputs[stage].append(dict(image=name,source=source,teacher=teacher,current=record['current'],label_sha256=row['label_sha256']))
    assert len(inputs['train'])==192 and len(inputs['inner'])==48 and len(inputs['outer'])==30
    (OUT/'config/Ultralytics').mkdir(parents=True);shutil.copy2('C:/Windows/Fonts/arial.ttf',OUT/'config/Ultralytics/Arial.ttf')
    save(OUT/'protocol.json',dict(pins=pins,runtime_fingerprint=runtime,new_checkpoint_sha256=digest,
        initialization_checkpoint_not_replaced=True,baseline='Frozen accepted three-model snapshot, not newer1280 plug experiment',
        all192_training_before_holdouts=True,net_tp_gain_training_and_inner_required=True,outer_nonregression=True,
        same_strict_teacher_crosssupport_and_two_view_rules=True,max_primary=5,max_extra=5,
        no_unmatched_increase_no_old_targets_lost_normal_zero=True,predictions_before_labels=True,
        current_gui_optional_cues_must_be_preserved_in_later_integration=True,
        fixed_last_checkpoint=True,validation_already_seen=True,no_automatic_deployment=True,reference_pending=True,field_accuracy=False))
    os.environ.update(YOLO_CONFIG_DIR=str(OUT/'config'),YOLO_OFFLINE='True',YOLO_AUTOINSTALL='False',HF_HUB_OFFLINE='1')
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(4);model=YOLO(str(weight));assert model.task=='segment' and dict(model.names)=={0:'unplugged_plug',1:'unplugged_jack'}
    class Capped:
        def predict(self,*args,**kw):
            torch.set_num_threads(4);result=model.predict(*args,**kw);torch.set_num_threads(4);return result
    capped=Capped();started=time.monotonic()
    try:
        for stage,cases in inputs.items():
            folder=OUT/stage;folder.mkdir();records=[];inferred=0;short_circuits=0
            for index,row in enumerate(cases):
                teacher,current=row['teacher'],row['current'];shape=teacher['predictions']['source_shape']
                room=len(current['all_predictions'])-len(current['primary'])<5
                support=any(p['confidence']>.25 for p in teacher['predictions']['merged_predictions'])
                if room and support:
                    image=read_image(row['source']);raw=predict(capped,image)
                    candidate=dict(image=row['image'],source_sha256=teacher['source_sha256'],weight_sha256=digest,predictions=raw,zoom_evidence=[])
                    if len(native_selection(candidate)['supplementary'])<5:candidate['zoom_evidence']=predict_seed_views(capped,image,recheck_proposals(raw))
                    inferred+=1
                else:
                    candidate=dict(image=row['image'],source_sha256=teacher['source_sha256'],weight_sha256=digest,
                        predictions=dict(source_shape=shape,merged_predictions=[],edge_kept_predictions=[]),zoom_evidence=[]);short_circuits+=1
                trial=append_backbone(teacher,current,candidate);assert trial['backbone_fallback_reason'] is None
                save(folder/(Path(row['image']).stem+'_predictions.json'),dict(image=row['image'],current=current,trial=trial,newcase=candidate))
                name=row['image'];label=DATA/'labels'/('val01' if stage=='outer' else 'train01')/(Path(name).stem+'.txt')
                assert sha(label)==row['label_sha256'];h,w=shape;targets=[]
                for line in label.read_text(encoding='utf-8').splitlines():
                    cls,cx,cy,bw,bh=map(float,line.split())
                    if cls in (3,4):targets.append(dict(class_id=int(cls)-3,box=[(cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h]))
                old,new=current['all_predictions'],trial['all_predictions'];oldhits,newhits=matches(old,targets)[0],matches(new,targets)[0]
                records.append(dict(image=name,metrics=dict(current=metric(old,targets),trial=metric(new,targets)),
                    gained=sorted(newhits-oldhits),lost=sorted(oldhits-newhits),additions=len(trial['backbone_additions']),label_sha256=sha(label)))
                save(OUT/'progress.json',dict(status='running',pid=os.getpid(),stage=stage,completed=index+1,total=len(cases),
                    inferred=inferred,exact_short_circuits=short_circuits,seconds=round(time.monotonic()-started,2)))
            totals={s:{k:sum(r['metrics'][s][k] for r in records) for k in ('tp','unmatched','fn','predictions','targets')} for s in ('current','trial')}
            gains=totals['trial']['tp']>totals['current']['tp'] if stage in ('train','inner') else totals['trial']['tp']>=totals['current']['tp']
            normal=sum(r['metrics']['trial']['predictions'] for r in records if r['image'].startswith('normal_'))
            passed=gains and totals['trial']['unmatched']<=totals['current']['unmatched'] and not any(r['lost'] for r in records) and normal==0
            assert {p:sha(Path(p)) for p in pins}==pins and residual_runtime_fingerprint(REPO)==runtime
            save(folder/'report.json',dict(status='complete',qualifies=passed,summary=totals,cases=records,normal_cues=normal,
                inferred=inferred,exact_short_circuits=short_circuits,reference_pending=True,field_accuracy=False,production_changed=False))
            if not passed:save(OUT/'progress.json',dict(status='rejected',stage=stage,summary=totals));return
        save(OUT/'progress.json',dict(status='source_only_pass_requires_live_reference',production_changed=False))
    except BaseException as error:
        save(OUT/'progress.json',dict(status='failed',error=type(error).__name__+': '+str(error)));raise
if __name__=='__main__':main()
