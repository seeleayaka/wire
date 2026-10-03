"""Second acceptance gate protects all current V3 cues, including1280 gains.

Run after the finite backbone workflow finishes. Reuse its hashed fresh source
predictions, infer missing holdouts only if this stricter current gate passes.
Never deploy or replace current weights automatically.
"""
import copy
import os
import shutil
import sys
import time
from pathlib import Path
sys.dont_write_bytecode=True
from current_port_baseline_audit import ROOT,REPO,BASE,DATA,load,save,read_current_case,read_targets
from inspection_agent.optional_port_crop_review import sha,read_image,predict
from inspection_agent.resolution_loose_plug_support import resolution_runtime_fingerprint
from inspection_agent.teacher_student_port_support import native_selection
from inspection_agent.context_port_recheck import recheck_proposals,predict_seed_views
from evaluate_full_backbone_acceptance import append_backbone
from audit_port_multiscale_acceptance import metric,matches

TRAIN=ROOT/'artifacts/port_full_backbone_20261003'
OUT=ROOT/'artifacts/backbone_current_policy_acceptance_20261003'


def main():
    if OUT.exists():raise FileExistsError('Preserve evidence; no duplicate gate')
    workflow=load(TRAIN/'pipeline_progress.json')
    assert workflow['status']=='complete','Wait for the original finite workflow; do not race caches'
    training=load(TRAIN/'full/report.json')
    assert training['status']=='complete' and training['completed_epochs']==2
    assert training['changed_early_backbone_tensors']>0 and training['frozen_parameters_changed']==0
    weight=Path(training['weights']['last.pt']['path']);digest=training['weights']['last.pt']['sha256']
    assert sha(weight)==digest
    frozen=resolution_runtime_fingerprint(REPO)
    assert frozen['accepted_feature']==training['runtime_fingerprint']
    pins={str(p):sha(p) for p in (Path(__file__),Path(__file__).with_name('current_port_baseline_audit.py'),
        Path(__file__).with_name('evaluate_full_backbone_acceptance.py'),Path(__file__).with_name('audit_port_multiscale_acceptance.py'),
        TRAIN/'full/report.json',TRAIN/'pipeline_progress.json',weight)}
    inputs={}
    for stage,count in (('train',192),('inner',48),('outer',30)):
        reportpath=BASE/stage/'report.json';pins[str(reportpath)]=sha(reportpath);report=load(reportpath)
        assert len(report['cases'])==count and report['qualifies']
        inputs[stage]=[]
        for entry in report['cases']:
            teacher,record=read_current_case(stage,entry,pins)
            cachepath=TRAIN/'evaluation'/stage/(Path(entry['image']).stem+'_predictions.json')
            candidate=None
            if cachepath.exists():
                cached=load(cachepath);pins[str(cachepath)]=sha(cachepath)
                assert cached['current']==record['current'], 'Frozen old baseline changed'
                candidate=cached['newcase']
                assert candidate['weight_sha256']==digest and candidate['source_sha256']==teacher['source_sha256']
                assert candidate['image']==entry['image'] and candidate['predictions']['source_shape']==teacher['predictions']['source_shape']
            inputs[stage].append(dict(entry=entry,teacher=teacher,current=record['trial'],candidate=candidate))
    (OUT/'config/Ultralytics').mkdir(parents=True);shutil.copy2('C:/Windows/Fonts/arial.ttf',OUT/'config/Ultralytics/Arial.ttf')
    save(OUT/'protocol.json',dict(pins=pins,runtime_fingerprint=frozen,new_checkpoint_sha256=digest,
        baseline='Current V3 resolution support277/344 training,63/80 inner,33/56 outer',
        all_current_cues_preserved=True,all192_train_before_holdouts=True,train_and_inner_net_gain_required=True,
        no_unmatched_increase=True,normal_zero=True,no_old_targets_lost=True,maximum_primary=5,maximum_extra=5,
        existing_thresholds_unchanged=True,old_teacher_crosssupport_unchanged=True,
        cached_new_weight_predictions_reused_sha_verified=True,missing_new_weight_predictions_inferred=True,
        validation_reused=True,no_automatic_deployment=True,reference_pending=True,field_accuracy=False))
    os.environ.update(YOLO_CONFIG_DIR=str(OUT/'config'),YOLO_OFFLINE='True',YOLO_AUTOINSTALL='False',HF_HUB_OFFLINE='1')
    started=time.monotonic();model=None
    try:
        for stage,rows in inputs.items():
            folder=OUT/stage;folder.mkdir();records=[];fresh=0;replayed=0;skipped=0
            for index,item in enumerate(rows):
                entry,teacher,current=item['entry'],item['teacher'],item['current'];name=entry['image']
                room=len(current['all_predictions'])-len(current['primary'])<5
                support=any(p['confidence']>.25 for p in teacher['predictions']['merged_predictions'])
                candidate=item['candidate']
                if candidate is not None:replayed+=1
                elif room and support:
                    if model is None:
                        import torch
                        from ultralytics import YOLO
                        torch.set_num_threads(4);native=YOLO(str(weight))
                        assert native.task=='segment' and dict(native.names)=={0:'unplugged_plug',1:'unplugged_jack'}
                        class Capped:
                            def predict(self,*a,**kw):
                                torch.set_num_threads(4);r=native.predict(*a,**kw);torch.set_num_threads(4);return r
                        model=Capped()
                    source=DATA/'images'/('val01' if stage=='outer' else 'train01')/name
                    image=read_image(source);raw=predict(model,image)
                    candidate=dict(image=name,source_sha256=teacher['source_sha256'],weight_sha256=digest,predictions=raw,zoom_evidence=[])
                    if len(native_selection(candidate)['supplementary'])<5:
                        candidate['zoom_evidence']=predict_seed_views(model,image,recheck_proposals(raw))
                    fresh+=1
                else:
                    candidate=dict(image=name,source_sha256=teacher['source_sha256'],weight_sha256=digest,zoom_evidence=[],
                                   predictions=dict(source_shape=teacher['predictions']['source_shape'],merged_predictions=[],edge_kept_predictions=[]))
                    skipped+=1
                trial=append_backbone(teacher,current,candidate)
                assert trial['backbone_fallback_reason'] is None
                assert trial['all_predictions'][:len(current['all_predictions'])]==current['all_predictions']
                save(folder/(Path(name).stem+'_predictions.json'),dict(image=name,current=current,trial=trial,newcase=candidate))
                targets=read_targets(stage,name,teacher['predictions']['source_shape'],entry['label_sha256'],pins)
                old,new=current['all_predictions'],trial['all_predictions'];oh,nh=matches(old,targets)[0],matches(new,targets)[0]
                records.append(dict(image=name,metrics=dict(current=metric(old,targets),trial=metric(new,targets)),
                                    gained=sorted(nh-oh),lost=sorted(oh-nh),additions=len(trial['backbone_additions'])))
                save(OUT/'progress.json',dict(status='running',pid=os.getpid(),stage=stage,completed=index+1,total=len(rows),
                     fresh=fresh,replayed=replayed,exact_short_circuits=skipped,seconds=round(time.monotonic()-started,2)))
            totals={v:{k:sum(r['metrics'][v][k] for r in records) for k in ('tp','unmatched','fn','predictions','targets')} for v in ('current','trial')}
            assert totals['current']==load(BASE/stage/'report.json')['summary']['trial']
            normal=sum(r['metrics']['trial']['predictions'] for r in records if r['image'].startswith('normal_'))
            gain=totals['trial']['tp']>totals['current']['tp'] if stage!='outer' else totals['trial']['tp']>=totals['current']['tp']
            passed=gain and totals['trial']['unmatched']<=totals['current']['unmatched'] and not any(r['lost'] for r in records) and normal==0
            assert {p:sha(Path(p)) for p in pins}==pins and resolution_runtime_fingerprint(REPO)==frozen
            save(folder/'report.json',dict(status='complete',qualifies=passed,summary=totals,cases=records,normal_cues=normal,
                 fresh=fresh,replayed=replayed,exact_short_circuits=skipped,source_only=True,reference_pending=True,field_accuracy=False))
            print(str(dict(stage=stage,qualifies=passed,summary=totals)),flush=True)
            if not passed:
                save(OUT/'progress.json',dict(status='rejected',stage=stage,summary=totals,no_automatic_deployment=True));return
        save(OUT/'progress.json',dict(status='current_source_gate_pass_requires_fresh_reference',no_automatic_deployment=True))
    except BaseException as error:
        save(OUT/'progress.json',dict(status='failed',error=type(error).__name__+': '+str(error)));raise


if __name__=='__main__':main()
