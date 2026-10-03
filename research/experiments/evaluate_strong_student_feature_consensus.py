"""Frozen strong-consensus experiment: train positives/controls, ALL192, holdouts.

Current V3 resolution cues are the baseline, not the older three-model state.
No production replacement, no thresholds changed after examining this result.
"""
import copy
import os
import shutil
import sys
import time
from pathlib import Path
sys.dont_write_bytecode=True
from current_port_baseline_audit import ROOT,REPO,BASE,EXTRA,DATA,load,save,read_current_case,read_targets
from inspection_agent.optional_port_crop_review import sha,read_image,predict
from inspection_agent.resolution_loose_plug_support import resolution_runtime_fingerprint
from inspection_agent.teacher_student_port_support import native_selection,STUDENT_SHA
from inspection_agent.feature_residual_port_support import WEIGHT_SHA,WEIGHT_RELATIVE
from inspection_agent.context_port_recheck import recheck_proposals,predict_seed_views
from strong_model_consensus_policy import append_strong_consensus,consistent_rows,POLICY
from audit_port_multiscale_acceptance import metric,matches

OUT=ROOT/'artifacts/strong_student_feature_consensus_20261003'
RESIDUAL=ROOT/'artifacts/port_residual_feature_support_20261003'
STUDENT=ROOT/'artifacts/port_training_multiscale_20261002/evaluation'
EXTENDED=ROOT/'artifacts/port_extended_training_controls_20261003'


def cached_cases(stage,entry,teacher,pins):
    name=entry['image'];stem=Path(name).stem
    if stage=='train' and (EXTRA/(stem+'_predictions.json')).exists():
        path=EXTRA/(stem+'_predictions.json');record=load(path)
        pins[str(path)]=sha(path)
        return record['student'],record['feature'],True
    if stage=='train' and (EXTENDED/(stem+'_predictions.json')).exists():
        path=EXTENDED/(stem+'_predictions.json');candidate=load(path)['versions']['student'];mode='extended'
    else:
        path=STUDENT/stage/(stem+'_zoom_predictions.json');candidate=load(path);mode=stage
    pins[str(path)]=sha(path)
    peerpath=RESIDUAL/mode/(stem+'_predictions.json');record=load(peerpath);pins[str(peerpath)]=sha(peerpath)
    peer=record['feature']
    # Train32 and inner48 feature predictions were always inferred. Other
    # historical caches were explicitly short-circuited by old teacher/slots.
    accepted=record['accepted']
    real=(mode in ('train','inner') or
          (any(p['confidence']>.25 for p in teacher['predictions']['merged_predictions']) and
           len(accepted['all_predictions'])-len(accepted['primary'])<5))
    return candidate,peer,real


def main():
    if OUT.exists():raise FileExistsError('Preserve previous trials')
    frozen=resolution_runtime_fingerprint(REPO)
    pins={str(p):sha(p) for p in (Path(__file__),Path(__file__).with_name('strong_model_consensus_policy.py'),
        Path(__file__).with_name('current_port_baseline_audit.py'),Path(__file__).with_name('audit_port_multiscale_acceptance.py'))}
    entries={stage:load(BASE/stage/'report.json')['cases'] for stage in ('train','inner','outer')}
    extended_names={r['image'] for r in load(RESIDUAL/'extended/report.json')['cases']}
    preflight=[e for e in entries['train'] if e['image'] not in extended_names]
    assert len(preflight)==56 and len(entries['train'])==192 and len(extended_names)==136
    # All cached inputs are pinned before any evaluation; labels are unopened.
    inputs={}
    for stage,rows in entries.items():
        inputs[stage]={}
        for entry in rows:
            teacher,record=read_current_case(stage,entry,pins)
            candidate,peer,real=cached_cases(stage,entry,teacher,pins)
            assert candidate['weight_sha256']==STUDENT_SHA and peer['weight_sha256']==WEIGHT_SHA
            assert candidate['source_sha256']==peer['source_sha256']==teacher['source_sha256']
            assert candidate['predictions']['source_shape']==peer['predictions']['source_shape']==teacher['predictions']['source_shape']
            inputs[stage][entry['image']]=dict(teacher=teacher,current=record['trial'],candidate=candidate,peer=peer,peer_real=real)
    (OUT/'config/Ultralytics').mkdir(parents=True)
    shutil.copy2('C:/Windows/Fonts/arial.ttf',OUT/'config/Ultralytics/Arial.ttf')
    save(OUT/'protocol.json',dict(policy=POLICY,pins=pins,runtime_fingerprint=frozen,
        checkpoints=dict(candidate=STUDENT_SHA,peer=WEIGHT_SHA),baseline='Current V3 loose-plug resolution277/63/33',
        stages=['all56_original_plus_remaining_controls','ALL192_training','inner48','outer30'],
        peer_cache_placeholder_never_treated_as_inferred_negative=True,
        exact_short_circuit='No strict candidate from student or current shared budget full makes peer unnecessary',
        class_agnostic=True,predictions_before_labels=True,validation_reused=True,
        require_net_training_and_inner_gain=True,no_unmatched_increase=True,no_old_target_loss=True,
        no_automatic_deployment=True,reference_pending=True,field_accuracy=False))
    os.environ.update(YOLO_CONFIG_DIR=str(OUT/'config'),YOLO_OFFLINE='True',YOLO_AUTOINSTALL='False',HF_HUB_OFFLINE='1')
    started=time.monotonic();fresh_count=0;model=None;source_cache={}
    phases=[('positive_preflight','train',preflight),('all_train','train',entries['train']),
            ('inner','inner',entries['inner']),('outer','outer',entries['outer'])]
    try:
        for phase,stage,rows in phases:
            folder=OUT/phase;folder.mkdir();records=[]
            for index,entry in enumerate(rows):
                name=entry['image'];item=inputs[stage][name];teacher,current=item['teacher'],item['current']
                cache_key=(stage,name)
                if cache_key in source_cache:
                    result=copy.deepcopy(source_cache[cache_key])
                else:
                    candidate,peer=item['candidate'],item['peer']
                    room=len(current['all_predictions'])-len(current['primary'])<5
                    if not item['peer_real'] and room and consistent_rows(candidate):
                        if model is None:
                            import torch
                            from ultralytics import YOLO
                            torch.set_num_threads(4);native=YOLO(str(REPO/WEIGHT_RELATIVE))
                            assert native.task=='segment' and dict(native.names)=={0:'unplugged_plug',1:'unplugged_jack'}
                            class Capped:
                                def predict(self,*a,**kw):
                                    torch.set_num_threads(4);r=native.predict(*a,**kw);torch.set_num_threads(4);return r
                            model=Capped()
                        source=DATA/'images'/('val01' if stage=='outer' else 'train01')/name
                        image=read_image(source);raw=predict(model,image)
                        peer=dict(image=name,source_sha256=teacher['source_sha256'],weight_sha256=WEIGHT_SHA,predictions=raw,zoom_evidence=[])
                        if len(native_selection(peer)['supplementary'])<5:
                            peer['zoom_evidence']=predict_seed_views(model,image,recheck_proposals(raw))
                        fresh_count+=1
                        save(folder/(Path(name).stem+'_fresh_peer.json'),peer)
                    result=append_strong_consensus(current,candidate,peer)
                    assert result['strong_consensus_fallback_reason'] is None
                    source_cache[cache_key]=copy.deepcopy(result)
                save(folder/(Path(name).stem+'_selection.json'),dict(image=name,current=current,trial=result))
                targets=read_targets(stage,name,teacher['predictions']['source_shape'],entry['label_sha256'],pins)
                old,new=current['all_predictions'],result['all_predictions'];oh,nh=matches(old,targets)[0],matches(new,targets)[0]
                records.append(dict(image=name,metrics=dict(current=metric(old,targets),trial=metric(new,targets)),
                                    gained=sorted(nh-oh),lost=sorted(oh-nh),additions=len(result['strong_consensus_additions'])))
                save(OUT/'progress.json',dict(status='running',phase=phase,completed=index+1,total=len(rows),
                     pid=os.getpid(),fresh_peer_sources=fresh_count,seconds=round(time.monotonic()-started,2)))
            totals={v:{k:sum(r['metrics'][v][k] for r in records) for k in ('tp','unmatched','fn','predictions','targets')} for v in ('current','trial')}
            normal=sum(r['metrics']['trial']['predictions'] for r in records if r['image'].startswith('normal_'))
            gain=totals['trial']['tp']>totals['current']['tp'] if phase!='outer' else totals['trial']['tp']>=totals['current']['tp']
            passed=gain and totals['trial']['unmatched']<=totals['current']['unmatched'] and normal==0 and not any(r['lost'] for r in records)
            assert {p:sha(Path(p)) for p in pins}==pins and resolution_runtime_fingerprint(REPO)==frozen
            save(folder/'report.json',dict(status='complete',qualifies=passed,summary=totals,cases=records,normal_cues=normal,
                 fresh_peer_sources=fresh_count,reference_pending=True,field_accuracy=False,production_changed=False))
            print(str(dict(phase=phase,qualifies=passed,summary=totals)),flush=True)
            if not passed:
                save(OUT/'progress.json',dict(status='rejected',phase=phase,summary=totals,no_automatic_deployment=True));return
        save(OUT/'progress.json',dict(status='source_only_pass_requires_fresh_reference',no_automatic_deployment=True))
    except BaseException as error:
        save(OUT/'progress.json',dict(status='failed',error=type(error).__name__+': '+str(error)));raise


if __name__=='__main__':main()
