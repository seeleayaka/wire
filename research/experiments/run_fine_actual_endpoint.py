"""TRAIN-only actual reference-filtering diagnosis of rejected fine candidates."""
import copy
import os
import shutil
import sys
import time
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,REPO,DATA,load,save,sha,read_image,REFERENCE_SHA
from current_port_baseline_audit import BASE,read_targets
from actual_port_native_rows import actual_native_rows
from paired_graph_hint_link import accepted_native_rows
from inspection_agent.paired_native_pose import native_pose_runtime_fingerprint,append_native_pose_review,HEAD_RELATIVE,HEAD_SHA
from inspection_agent.paired_median_geometry import run_paired_median_review
from inspection_agent.paired_port_features import expected_in_source,valid_boxes,embeddings,paired_features
from inspection_agent.optional_port_crop_review import SCENE,predict,aligned_predictions
from inspection_agent.teacher_student_port_support import TEACHER_SHA,TEACHER_RELATIVE,STUDENT_SHA,STUDENT_RELATIVE,append_verified_student
from inspection_agent.context_port_recheck import predict_seed_views,complete
from inspection_agent.feature_residual_port_support import WEIGHT_SHA
from raw_pose_consensus import proposals,select
from audit_port_multiscale_acceptance import metric,matches
OUT=ROOT/'artifacts/fine_actual_endpoint_20261004'
PLAN=ROOT/'artifacts/fine_actual_endpoint_preregistration_20261004/PLAN.md'
FINE=ROOT/'artifacts/paired_fine_tile_views_20261003/full'
PRIOR=ROOT/'artifacts/fine_native_consensus_20261004'
os.environ.update(QT_QPA_PLATFORM='offscreen',HF_HUB_OFFLINE='1',YOLO_OFFLINE='True',
    YOLO_AUTOINSTALL='False',YOLO_CONFIG_DIR=str(OUT/'config'),PYTHONPROJECT10_DINO_CACHE=str(OUT/'dino_cache'))


def main():
    if OUT.exists():raise FileExistsError('Preserve fine actual endpoint diagnostic')
    import psutil
    assert psutil.virtual_memory().available>6*2**30,'No competing memory-heavy model job'
    import torch
    import cv2
    import numpy as np
    torch.set_num_threads(2);cv2.setNumThreads(1)
    OUT.mkdir();(OUT/'config/Ultralytics').mkdir(parents=True)
    shutil.copy2('C:/Windows/Fonts/arial.ttf',OUT/'config/Ultralytics/Arial.ttf')
    from ultralytics import YOLO
    import assembly_auto_review_dino_v2 as entry
    import dino_feature_diff as dino
    gui=entry.implementation;app=gui.QApplication.instance() or gui.QApplication([])
    frozen=native_pose_runtime_fingerprint(REPO)
    prior=load(PRIOR/'report.json');assert prior['status']=='rejected'
    assert native_pose_runtime_fingerprint(REPO)==prior['runtime']
    priorrows=load(PRIOR/'train/report.json')['cases']
    changed=sorted(r['image'] for r in priorrows if r['trial']['predictions']>r['current']['predictions'])
    oldunmatched=sorted(r['image'] for r in priorrows if r['current']['unmatched'])
    groupspath=ROOT/'artifacts/port_training_multiscale_20261002/protocol.json'
    normal=sorted(n for n in load(groupspath)['train_sources'] if n.startswith('normal_'))[:3]
    names=sorted(set(changed+oldunmatched+normal));assert len(changed)==8 and len(oldunmatched)==4 and len(names)==15
    entries={r['image']:r for r in load(BASE/'train/report.json')['cases']}
    referencepath=DATA/'images/train01/normal_073.JPG';assert sha(referencepath)==REFERENCE_SHA
    pins={str(p):sha(p) for p in (Path(__file__),PLAN,groupspath,referencepath,PRIOR/'report.json',
        PRIOR/'train/report.json',FINE/'report.json',REPO/HEAD_RELATIVE,
        Path(__file__).with_name('raw_pose_consensus.py'),Path(__file__).with_name('actual_port_native_rows.py'),
        Path(__file__).with_name('paired_graph_hint_link.py'))}
    assert pins[str(REPO/HEAD_RELATIVE)]==HEAD_SHA
    started=time.monotonic();cases=[]
    save(OUT/'protocol.json',dict(pins=pins,runtime=frozen,changed_sources=changed,
        all_baseline_unmatched_sources=oldunmatched,normal_controls=normal,selected_train_sources=names,
        diagnostic_only=True,old_source_failure_not_revoked=True,no_validation_read=True,
        no_training=True,no_SAM_recomputation=True,no_deployment=True,field_accuracy=False))
    head=torch.nn.Linear(6144,3);head.load_state_dict(torch.load(REPO/HEAD_RELATIVE,map_location='cpu',weights_only=True)['state_dict'])
    head.eval().requires_grad_(False);encoder=None
    reference=read_image(referencepath);previous=gui.adaptive.robust.auto.base.OUT
    try:
        for i,name in enumerate(names):
            source=DATA/'images/train01'/name;pins[str(source)]=sha(source)
            folder=OUT/Path(name).stem;folder.mkdir();gui.adaptive.robust.auto.base.OUT=folder
            def progress(phase):save(OUT/'progress.json',dict(status='running',pid=os.getpid(),image=name,
                completed=i,total=15,phase=phase,seconds=round(time.monotonic()-started,2)))
            progress('fresh_original_SIFT_DINO_then_accepted_baseline');cv2.setRNGSeed(0)
            worker=gui.InitialReviewWorker(referencepath,source,[[.03,.04,.97,.96]]);payloads=[]
            worker.completed.connect(payloads.append);worker.run()
            assert len(payloads)==1 and payloads[0]['status']=='ready_for_sam3',payloads[0].get('error') if payloads else 'no payload'
            report=payloads[0]['report'];directory=Path(payloads[0]['output']);save(directory/'initial_report.json',report)
            protected=copy.deepcopy(report)
            median=run_paired_median_review(report,project=REPO,median_enabled=True,paired_enabled=True,enabled=True,
                scene=SCENE,supplementary_enabled=True,student_enabled=True,feature_enabled=True,resolution_enabled=True)
            baseline=append_native_pose_review(report,median,project=REPO)
            oldrows=actual_native_rows(report,baseline)
            save(directory/'baseline_native_ports.json',baseline)
            trial=copy.deepcopy(baseline);proposed=[];scores=[];selected=[];added=[];reference_evidence=[];fineviews=[];reason=None
            if baseline['status']=='applied' and len(baseline.get('supplementary_hints',[]))<5:
                evidence=baseline['teacher_student_evidence'];teacher=evidence['teacher_case']
                pool=[teacher,evidence['student_case'],baseline['feature_residual_evidence']['feature_case'],baseline['resolution_evidence']['alternative']]
                finepath=FINE/'train'/(Path(name).stem+'_predictions.json');pins[str(finepath)]=sha(finepath)
                fine=load(finepath);fineviews=fine['new_views']
                for view in fineviews:assert view['source_sha256']==pins[str(source)] and view['weight_sha256'] in (TEACHER_SHA,STUDENT_SHA)
                current=dict(primary=oldrows[:len(baseline['rescue_hints'])],all_predictions=copy.deepcopy(oldrows))
                proposed=proposals(teacher,pool+fineviews,current)
                image=read_image(source);matrix=np.asarray(report['alignment']['source_to_reference_homography'],dtype=np.float64)
                assert report['alignment']['alignment_quality']['reliable']
                expected,mask=expected_in_source(reference,matrix,image.shape[:2])
                proposed=[proposed[j] for j in valid_boxes([r['box_xyxy'] for r in proposed],mask)]
                if proposed:
                    if encoder is None:encoder=dino._model();encoder.eval().requires_grad_(False)
                    torch.set_num_threads(2);boxes=[r['box_xyxy'] for r in proposed]
                    vectors=paired_features(embeddings(encoder,image,boxes),embeddings(encoder,expected,boxes))
                    with torch.inference_mode():scores=head(vectors).softmax(dim=1).tolist()
                    selected=select(current,proposed,scores,HEAD_SHA)['paired_semantic_additions']
                refs=[]
                if selected:
                    progress('fresh_normal_reference_and_candidate_ROI_checks')
                    candidates=copy.deepcopy(selected)
                    for candidate in candidates:candidate['support_tiles']=[]
                    mapped=aligned_predictions(candidates,matrix,image.shape[:2],reference.shape[:2])
                    seeds=[dict(class_id=p['class_id'],confidence=p['confidence'],box_xyxy=[p[k] for k in ('left','top','right','bottom')]) for p in mapped]
                    for digest,relative in ((TEACHER_SHA,TEACHER_RELATIVE),(STUDENT_SHA,STUDENT_RELATIVE)):
                        assert sha(REPO/relative)==digest;model=YOLO(str(REPO/relative))
                        assert model.task=='segment' and dict(model.names)=={0:'unplugged_plug',1:'unplugged_jack'}
                        class Capped:
                            def predict(self,*args,**kw):
                                torch.set_num_threads(2);result=model.predict(*args,**kw);torch.set_num_threads(2);return result
                        wrapped=Capped();raw=predict(wrapped,reference);views=predict_seed_views(wrapped,reference,seeds)
                        refs.extend(aligned_predictions(raw['merged_predictions'],np.eye(3),reference.shape[:2],reference.shape[:2]))
                        for record in views:
                            for view in record['views']:
                                for row in view:
                                    if row['confidence']>.25 and complete(row,reference.shape[:2]):
                                        refs.append(dict(zip(('left','top','right','bottom'),row['box_xyxy']),class_id=row['class_id'],
                                            confidence=row['confidence'],valid_warp_fraction=1.,support_tiles=[]))
                        reference_evidence.append(dict(weight_sha256=digest,predictions=raw,views=views))
                    trial,added=append_verified_student(baseline,selected,refs,matrix)
                    newrows=accepted_native_rows(selected,added,matrix,image.shape[:2],reference.shape[:2])
                else:newrows=[]
            else:newrows=[];reason='baseline_abstention_or_shared_budget_full'
            assert report==protected
            for key,value in baseline.items():
                if key=='supplementary_hints':assert trial[key][:len(value)]==value
                else:assert trial[key]==value
            for hint in added:hint.update(evidence_tier='fine_actual_endpoint_diagnostic_only',
                automatic_fault_verdict=False,paired_geometry_head_sha256=HEAD_SHA,
                warning='Rejected source branch under TRAIN-only endpoint diagnosis; no physical fault verdict.')
            save(directory/'fine_actual_ports.json',trial)
            save(directory/'fine_diagnostic_evidence.json',dict(proposals=proposed,probabilities=scores,selected=selected,
                new_actual_rows=newrows,reference_evidence=reference_evidence,cached_fine_views=fineviews,reason=reason))
            targets=read_targets('train',name,[2736,3648],entries[name]['label_sha256'],pins)
            oldhits=matches(oldrows,targets)[0];newhits=matches(oldrows+newrows,targets)[0]
            row=dict(image=name,baseline=metric(oldrows,targets),trial=metric(oldrows+newrows,targets),
                selected=len(selected),added=len(newrows),gained=sorted(newhits-oldhits),lost=sorted(oldhits-newhits),
                safety_abstention=baseline['status']!='applied',reason=reason,
                report=str(directory/'initial_report.json'),baseline_result=str(directory/'baseline_native_ports.json'),
                trial_result=str(directory/'fine_actual_ports.json'),evidence=str(directory/'fine_diagnostic_evidence.json'))
            cases.append(row);save(OUT/'partial.json',dict(cases=cases));print(dict(completed=i+1,total=15,**row),flush=True)
            assert native_pose_runtime_fingerprint(REPO)==frozen and sha(source)==pins[str(source)]
        totals={version:{k:sum(r[version][k] for r in cases) for k in ('tp','unmatched','fn','predictions','targets')} for version in ('baseline','trial')}
        normal_added=sum(r['added'] for r in cases if r['image'] in normal)
        qualifies=totals['trial']['tp']>totals['baseline']['tp'] and totals['trial']['unmatched']<=totals['baseline']['unmatched'] and normal_added==0 and not any(r['lost'] for r in cases)
        assert all(sha(Path(p))==v for p,v in pins.items()) and native_pose_runtime_fingerprint(REPO)==frozen
        result=dict(status='diagnostic_net_gain_requires_full_cohort_and_holdouts' if qualifies else 'rejected',
            qualifies=qualifies,summary=totals,cases=cases,pins=pins,runtime=frozen,
            normal_added=normal_added,selected_train_only_not_population_accuracy=True,
            old_source_failure_not_revoked=True,no_training=True,no_SAM_recomputation=True,
            no_deployment=True,field_accuracy=False,seconds=round(time.monotonic()-started,2))
        save(OUT/'report.json',result);save(OUT/'progress.json',dict(status=result['status'],seconds=result['seconds'],summary=totals))
        print(dict(status=result['status'],summary=totals),flush=True)
    except BaseException as error:
        save(OUT/'progress.json',dict(status='failed',error=type(error).__name__+': '+str(error)));raise
    finally:gui.adaptive.robust.auto.base.OUT=previous


if __name__=='__main__':main()
