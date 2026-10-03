"""Every prospective graph addition gets fresh workflow and reference gating."""
import copy
import json
import os
import sys
import time
from pathlib import Path
sys.dont_write_bytecode=True
from current_port_baseline_audit import ROOT,REPO,BASE,DATA,load,read_targets
sys.path[:0]=[str(REPO),str(REPO/'prototype')]
OUT=ROOT/'artifacts/paired_support_graph_20261003'
os.environ.update(QT_QPA_PLATFORM='offscreen',HF_HUB_OFFLINE='1',YOLO_OFFLINE='True',YOLO_AUTOINSTALL='False',
    YOLO_CONFIG_DIR=str(OUT/'config'),PYTHONPROJECT10_DINO_CACHE=str(OUT/'dino_cache'))
from inspection_agent.optional_port_crop_review import sha,read_image,predict,aligned_predictions,SCENE,REFERENCE_SHA
from inspection_agent.resolution_loose_plug_support import resolution_runtime_fingerprint,run_resolution_plug_review
from inspection_agent.teacher_student_port_support import STUDENT_SHA,STUDENT_RELATIVE,native_selection,append_verified_student
from inspection_agent.feature_residual_port_support import WEIGHT_SHA,WEIGHT_RELATIVE
from inspection_agent.context_port_recheck import predict_seed_views,recheck_proposals,complete,render_consensus_overlay
from port_support_graph_policy import append_support_graph,POLICY
from paired_graph_hint_link import accepted_native_rows
from audit_port_multiscale_acceptance import metric,matches


def save(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');tmp.replace(path)


def assert_parity(cached,fresh):
    import numpy as np
    if len(cached)!=len(fresh):raise ValueError('fresh_graph_count_mismatch')
    for a,b in zip(cached,fresh):
        if a['class_id']!=b['class_id'] or abs(a['confidence']-b['confidence'])>1e-6:
            raise ValueError('fresh_graph_class_score_mismatch')
        if not np.allclose(a['box_xyxy'],b['box_xyxy'],rtol=0,atol=.001):raise ValueError('fresh_graph_geometry_mismatch')


def main():
    destination=OUT/'paired_acceptance'
    if destination.exists():raise FileExistsError('Preserve prior paired evidence')
    prepared=load(OUT/'source_selections/protocol.json');assert prepared['status']=='complete'
    pins=dict(prepared['pins']);frozen=resolution_runtime_fingerprint(REPO);assert frozen==prepared['runtime_fingerprint']
    reference=DATA/'images/train01/normal_073.JPG';assert sha(reference)==REFERENCE_SHA
    for p in (Path(__file__),Path(__file__).with_name('paired_graph_hint_link.py'),
              Path(__file__).with_name('port_support_graph_policy.py'),OUT/'source_selections/protocol.json',reference):pins[str(p)]=sha(p)
    inputs={}
    for stage,count in (('train',192),('inner',48),('outer',30)):
        indexpath=OUT/'source_selections'/stage/'index.json';pins[str(indexpath)]=sha(indexpath);items=load(indexpath)['records'];assert len(items)==count
        for row in items:assert sha(Path(row['path']))==row['sha256'];pins[row['path']]=row['sha256']
        inputs[stage]=items
    assert {p:sha(Path(p)) for p in pins}==pins
    destination.mkdir()
    save(destination/'protocol.json',dict(pins=pins,runtime_fingerprint=frozen,policy=POLICY,reference_sha256=REFERENCE_SHA,
        fixed_roi=[.03,.04,.97,.96],fresh_registration_DINO_on_all_candidate_sources=True,
        fresh_source_graph_parity_on_all_candidate_sources=True,source_cache_parity_tolerance_pixels=.001,confidence_tolerance=1e-6,
        fresh_reference_full_both_checkpoints_and_matched_crops=True,exact_native_hint_link_no_inverse_warp_scoring=True,
        current_source_prefix_and_live_hints_preserved=True,source_no_candidate_short_circuit_exact=True,
        no_ground_truth_inputs_to_policy=True,no_automatic_deployment=True,validation_reused=True,field_accuracy=False))
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO
    import assembly_auto_review_dino_v2 as entry
    torch.set_num_threads(4);gui=entry.implementation;app=gui.QApplication.instance() or gui.QApplication([])
    previous=gui.adaptive.robust.auto.base.OUT;started=time.monotonic();models={};reference_cache={};refimage=read_image(reference)
    def capped(digest,relative):
        if digest not in models:
            native=YOLO(str(REPO/relative));assert sha(REPO/relative)==digest
            assert native.task=='segment' and dict(native.names)=={0:'unplugged_plug',1:'unplugged_jack'}
            class Capped:
                def predict(self,*a,**kw):
                    torch.set_num_threads(4);value=native.predict(*a,**kw);torch.set_num_threads(4);return value
            models[digest]=Capped()
        return models[digest]
    def fresh_source(model,digest,image,name,directory):
        raw=predict(model,image);case=dict(image=name,source_sha256=sha(DATA/'images'/('val01' if stage=='outer' else 'train01')/name),
            weight_sha256=digest,predictions=raw,zoom_evidence=[])
        if len(native_selection(case)['supplementary'])<5:case['zoom_evidence']=predict_seed_views(model,image,recheck_proposals(raw))
        save(directory/(digest[:8]+'_fresh_source.json'),case);return case
    def progress(**kw):save(destination/'progress.json',dict(status='running',pid=os.getpid(),seconds=round(time.monotonic()-started,2),**kw))
    try:
        for stage,items in inputs.items():
            folder=destination/stage;folder.mkdir();records=[];livecount=0
            for index,row in enumerate(items):
                case=load(Path(row['path']));name=row['image'];current=case['current'];candidates=case['trial']['strong_consensus_additions'];accepted=[]
                live=None
                if candidates:
                    target=folder/Path(name).stem;target.mkdir();gui.adaptive.robust.auto.base.OUT=target
                    progress(stage=stage,image=name,completed=index,total=len(items),phase='fresh_initial_registration_DINO')
                    source=DATA/'images'/('val01' if stage=='outer' else 'train01')/name
                    cv2.setRNGSeed(0);worker=gui.InitialReviewWorker(reference,source,[[.03,.04,.97,.96]])
                    payloads=[];worker.completed.connect(payloads.append);worker.run();assert len(payloads)==1
                    payload=payloads[0];assert payload['status']=='ready_for_sam3',payload['status']
                    report=payload['report'];directory=Path(payload['output']);save(directory/'initial_report.json',report)
                    untouched=copy.deepcopy(report)
                    progress(stage=stage,image=name,completed=index,total=len(items),phase='current_V3_live_ports')
                    original=run_resolution_plug_review(report,project=REPO,enabled=True,scene=SCENE,supplementary_enabled=True,
                        student_enabled=True,feature_enabled=True,resolution_enabled=True)
                    assert report==untouched
                    assert original['status']=='applied',original.get('fallback_reason')
                    for key in ('teacher_student_policy','feature_residual_policy','resolution_policy'):
                        assert not original.get(key,{}).get('fallback_reason'),original.get(key)
                    save(directory/'current_v3_ports.json',original)
                    image=read_image(source);source_sha=sha(source);assert source_sha==case['student']['source_sha256']==case['feature']['source_sha256']
                    student_model=capped(STUDENT_SHA,STUDENT_RELATIVE);feature_model=capped(WEIGHT_SHA,WEIGHT_RELATIVE)
                    progress(stage=stage,image=name,completed=index,total=len(items),phase='fresh_graph_source_parity')
                    student=original.get('teacher_student_evidence',{}).get('student_case')
                    feature=original.get('feature_residual_evidence',{}).get('feature_case')
                    if student is None:student=fresh_source(student_model,STUDENT_SHA,image,name,directory)
                    if feature is None:feature=fresh_source(feature_model,WEIGHT_SHA,image,name,directory)
                    assert student['source_sha256']==feature['source_sha256']==source_sha
                    fresh_graph=append_support_graph(current,student,feature);assert fresh_graph['strong_consensus_fallback_reason'] is None
                    fresh_candidates=fresh_graph['strong_consensus_additions'];assert_parity(candidates,fresh_candidates)
                    matrix=np.asarray(report['alignment']['source_to_reference_homography'],dtype=np.float64)
                    native=copy.deepcopy(fresh_candidates)
                    for p in native:p['support_tiles']=[]
                    mapped=aligned_predictions(native,matrix,image.shape[:2],refimage.shape[:2])
                    seeds=[dict(class_id=p['class_id'],confidence=p['confidence'],box_xyxy=[p[k] for k in ('left','top','right','bottom')]) for p in mapped]
                    progress(stage=stage,image=name,completed=index,total=len(items),phase='fresh_both_reference_checkpoints_and_crops')
                    refs=[];reference_evidence=[]
                    for digest,model in ((STUDENT_SHA,student_model),(WEIGHT_SHA,feature_model)):
                        if digest not in reference_cache:
                            raw=predict(model,refimage);reference_cache[digest]=raw
                            save(destination/(digest[:8]+'_reference_full.json'),dict(reference_sha256=REFERENCE_SHA,weight_sha256=digest,predictions=raw))
                        raw=reference_cache[digest];refs.extend(aligned_predictions(raw['merged_predictions'],np.eye(3),refimage.shape[:2],refimage.shape[:2]))
                        views=predict_seed_views(model,refimage,seeds);reference_evidence.append(dict(weight_sha256=digest,views=views))
                        for record in views:
                            for view in record['views']:
                                for p in view:
                                    if p['confidence']>.25 and complete(p,refimage.shape[:2]):
                                        refs.append(dict(zip(('left','top','right','bottom'),p['box_xyxy']),class_id=p['class_id'],
                                            confidence=p['confidence'],valid_warp_fraction=1.,support_tiles=[]))
                    output,hints=append_verified_student(original,fresh_candidates,refs,matrix)
                    assert output['parents']==original['parents'] and output['existing_hints']==original['existing_hints'] and output['rescue_hints']==original['rescue_hints']
                    assert output['supplementary_hints'][:len(original['supplementary_hints'])]==original['supplementary_hints']
                    accepted=accepted_native_rows(fresh_candidates,hints,matrix,image.shape[:2],refimage.shape[:2])
                    assert len(accepted)<=len(candidates)
                    save(directory/'paired_graph_evidence.json',dict(fresh_graph=fresh_graph,reference_rows=refs,matched_reference_evidence=reference_evidence,
                        current_ports=original,trial_ports=output,accepted_native=accepted,graph_hints=hints,source_parity_passed=True,
                        sam_pending=True,automatic_fault_verdict=False))
                    render_consensus_overlay(directory/'aligned.jpg',output,directory/'paired_graph_overlay.jpg')
                    live=dict(initial_report=str(directory/'initial_report.json'),evidence=str(directory/'paired_graph_evidence.json'),
                        overlay=str(directory/'paired_graph_overlay.jpg'),native_candidates=len(candidates),accepted=len(accepted),
                        status='applied',sam_pending=True,decision=output['decision'])
                    livecount+=1
                selected=copy.deepcopy(current['all_predictions'])+accepted
                assert selected[:len(current['all_predictions'])]==current['all_predictions'] and len(selected)<=len(current['primary'])+5
                targets=read_targets(stage,name,case['teacher']['predictions']['source_shape'],case['entry']['label_sha256'],pins)
                old=current['all_predictions'];oh,nh=matches(old,targets)[0],matches(selected,targets)[0]
                records.append(dict(image=name,current=metric(old,targets),trial=metric(selected,targets),gained=sorted(nh-oh),lost=sorted(oh-nh),
                    native_candidates=len(candidates),accepted=len(accepted),paired_live=live))
                save(folder/(Path(name).stem+'_scored.json'),dict(image=name,current=current,trial_native=selected,targets=targets,record=records[-1]))
                save(folder/'partial.json',dict(cases=records));progress(stage=stage,image=name,completed=index+1,total=len(items),phase='paired_source_scoring',fresh_candidate_sources=livecount)
            totals={v:{k:sum(r[v][k] for r in records) for k in ('tp','unmatched','fn','predictions','targets')} for v in ('current','trial')}
            assert totals['current']==load(BASE/stage/'report.json')['summary']['trial']
            normal=sum(r['trial']['predictions'] for r in records if r['image'].startswith('normal_'))
            gain=totals['trial']['tp']>totals['current']['tp'] if stage!='outer' else totals['trial']['tp']>=totals['current']['tp']
            passed=gain and totals['trial']['unmatched']<=totals['current']['unmatched'] and normal==0 and not any(r['lost'] for r in records)
            assert {p:sha(Path(p)) for p in pins}==pins and resolution_runtime_fingerprint(REPO)==frozen
            save(folder/'report.json',dict(status='complete',qualifies=passed,summary=totals,cases=records,normal_cues=normal,
                fresh_all_candidate_sources=livecount,old_baseline_source_cached=True,no_candidate_sources_not_fresh_inferred=True,
                validation_reused=True,sam_pending=True,field_accuracy=False,production_changed=False))
            print(str(dict(stage=stage,qualifies=passed,summary=totals)),flush=True)
            if not passed:save(destination/'progress.json',dict(status='rejected',stage=stage,summary=totals));return
        save(destination/'progress.json',dict(status='paired_source_pass_requires_new_GUI_SAM_acceptance',no_automatic_deployment=True))
    except BaseException as error:
        save(destination/'progress.json',dict(status='failed',error=type(error).__name__+': '+str(error),seconds=round(time.monotonic()-started,2)));raise
    finally:
        gui.adaptive.robust.auto.base.OUT=previous;app.processEvents()


if __name__=='__main__':main()
