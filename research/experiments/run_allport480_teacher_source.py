"""Conditional new teacher role; frozen TRAIN source cascade + strict gates."""
import os
import sys
import time
import copy
import shutil
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,REPO,DATA,load,save,sha,read_image,REFERENCE_SHA
from current_port_baseline_audit import BASE,read_targets
from run_allport480_source import NEW_SHA,FINE,PAIR,cached_view,full_frame
from inspection_agent.paired_native_pose import native_pose_runtime_fingerprint,HEAD_SHA,HEAD_RELATIVE
from inspection_agent.paired_port_features import expected_in_source,valid_boxes,embeddings,paired_features
from inspection_agent.optional_port_crop_review import CONFIG,predict
from inspection_agent.teacher_student_port_support import STUDENT_SHA,STUDENT_RELATIVE
from inspection_agent.feature_residual_port_support import WEIGHT_SHA
from paired_fine_tile_views import infer as fine_infer
from replacement_voters import validate_views,check_prefix
from raw_pose_consensus import proposals
from fine_voter_quality import attach
from consensus_rank import select
from audit_port_multiscale_acceptance import matches,metric
from exact_paired_score_cache import binding as semantic_binding,reuse as semantic_reuse
from protected_allport_baseline import check_accepted,check_totals

SOURCE=ROOT/'artifacts/allport480_source_20261005'
AUDIT=ROOT/'artifacts/allport480_source_audit_20261005/report.json'
OUT=ROOT/'artifacts/allport480_teacher_source_20261005'
PLAN=ROOT/'artifacts/allport480_teacher_source_preregistration_20261005/PLAN.md'
ACCEPTED=ROOT/'artifacts/paired_pose_native_three_20261004'


def main():
    if OUT.exists():raise FileExistsError('preserve teacher replacement source experiment')
    prior=load(SOURCE/'report.json');audit=load(AUDIT)
    if prior['status']!='rejected' or prior['qualifies'] or audit['status']!='pass' or audit['candidate_source_qualifies']:
        raise ValueError('only after complete independently rejected first source trial')
    if audit['source_report_sha256']!=sha(SOURCE/'report.json') or any(sha(p)!=d for p,d in prior['pins'].items()):
        raise ValueError('first trial provenance drift')
    audit_cases=audit.get('source_case_sha256',{})
    if (len(audit_cases)!=192 or audit['source_protocol_sha256']!=sha(SOURCE/'protocol.json')
        or any(sha(p)!=d for p,d in audit_cases.items())):raise ValueError('independently audited source outputs drift')
    frozen=native_pose_runtime_fingerprint(REPO)
    if prior['runtime']!=frozen:raise ValueError('mainline drift')
    protocol=load(SOURCE/'protocol.json');original_config=copy.deepcopy(CONFIG)
    fine_protocol=load(FINE/'protocol.json')
    if protocol['original_config']!=CONFIG or fine_protocol['original_config']!=CONFIG or any(sha(p)!=d for p,d in fine_protocol['pins'].items()):
        raise ValueError('cached view/code/options mismatch')
    import psutil,cv2,torch
    if psutil.virtual_memory().available<6*2**30:raise RuntimeError('do not compete for model memory')
    torch.set_num_threads(2);cv2.setNumThreads(1)
    roles=dict(teacher=NEW_SHA,student=STUDENT_SHA,feature=WEIGHT_SHA)
    pins={str(p):sha(p) for p in [Path(__file__),PLAN,SOURCE/'report.json',SOURCE/'protocol.json',AUDIT,PAIR,FINE/'protocol.json',FINE/'report.json',
        REPO/STUDENT_RELATIVE,REPO/HEAD_RELATIVE,ACCEPTED/'report.json',*[Path(__file__).with_name(n+'.py') for n in ('run_allport480_source','protected_allport_baseline','replacement_voters','exact_paired_score_cache','raw_pose_consensus','fine_voter_quality','consensus_rank','novel_box_geometry','relative_port_box','paired_fine_tile_views','prepare_paired_port_semantics','current_port_baseline_audit','audit_port_multiscale_acceptance')]]}
    if pins[str(REPO/STUDENT_RELATIVE)]!=STUDENT_SHA or pins[str(REPO/HEAD_RELATIVE)]!=HEAD_SHA:raise ValueError('weight/head drift')
    referencepath=DATA/'images/train01/normal_073.JPG'
    if sha(referencepath)!=REFERENCE_SHA:raise ValueError('reference changed')
    pins[str(referencepath)]=REFERENCE_SHA
    dirty=lambda:__import__('subprocess').check_output(['E:/Git/cmd/git.exe','-C',str(REPO),'status','--porcelain'],text=True)
    before_dirty=dirty();indexed={r['image']:r for r in load(PAIR)['records']};entries=load(BASE/'train/report.json')['cases']
    if len(entries)!=192 or sorted(r['image'] for r in entries)!=sorted(protocol['train_sources']):raise ValueError('TRAIN membership changed')
    OUT.mkdir();(OUT/'config/Ultralytics').mkdir(parents=True);shutil.copy2('C:/Windows/Fonts/arial.ttf',OUT/'config/Ultralytics/Arial.ttf')
    os.environ.update(YOLO_CONFIG_DIR=str(OUT/'config'),YOLO_OFFLINE='True',YOLO_AUTOINSTALL='False',HF_HUB_OFFLINE='1')
    from ultralytics import YOLO
    model=YOLO(str(REPO/STUDENT_RELATIVE))
    if model.task!='segment' or dict(model.names)!={0:'unplugged_plug',1:'unplugged_jack'}:raise ValueError('student class contract failed')
    class Capped:
        def predict(self,*a,**kw):
            torch.set_num_threads(2);out=model.predict(*a,**kw);torch.set_num_threads(2);return out
    student=Capped();head=torch.nn.Linear(6144,3)
    head.load_state_dict(torch.load(REPO/HEAD_RELATIVE,map_location='cpu',weights_only=True)['state_dict']);head.eval().requires_grad_(False)
    encoder=None;reference=read_image(referencepath);started=time.monotonic();rows=[];fresh=0;candidate_count=0;reused_scores=0;fresh_scores=0
    folder=OUT/'train';folder.mkdir()
    save(OUT/'protocol.json',dict(pins=pins,runtime=frozen,roles=roles,original_config=original_config,train_sources=protocol['train_sources'],
        mainline_dirty_before=before_dirty,wall_bound_seconds=7200,head_sha256=HEAD_SHA,protected_prefixes=[295,298],
        no_heldout_reads=True,no_deployment=True,same_weight_one_vote=True,GT_after_predictions=True,
        exact_semantic_cache_only=True,cache_is_not_an_extra_vote=True,approximate_semantic_cache_hits=False))
    try:
        for index,entry in enumerate(entries):
            if time.monotonic()-started>7200:raise TimeoutError('operational timeout, partial is not source acceptance')
            name=entry['image'];stem=Path(name).stem
            progress=lambda phase:save(OUT/'progress.json',dict(status='running',pid=os.getpid(),image=name,completed=index,total=192,phase=phase,
                fresh_view_calls=fresh,proposals=candidate_count,reused_semantic_scores=reused_scores,
                fresh_semantic_scores=fresh_scores,seconds=round(time.monotonic()-started,2)))
            progress('prepare_verified_source')
            casepath=SOURCE/'train'/(stem+'_predictions.json');pins[str(casepath)]=sha(casepath);case=load(casepath)
            if pins[str(casepath)]!=audit_cases[str(casepath)]:raise ValueError('audited source case changed before reuse')
            source=DATA/'images/train01'/name;pins[str(source)]=sha(source)
            if pins[str(source)]!=case['source_sha256'] or pins[str(source)]!=prior['pins'][str(source)]:raise ValueError('source pixels changed')
            acceptedpath=ACCEPTED/'full_train'/(stem+'_predictions.json');pins[str(acceptedpath)]=sha(acceptedpath)
            original=load(acceptedpath)['trial'];current=case['current'];check_accepted(case['original'],original,current,current)
            if prior.get('aggregation_recovery') and prior['pins'].get(str(acceptedpath))!=pins[str(acceptedpath)]:raise ValueError('accepted295 source baseline drift')
            alignment=case['alignment'];eligible=case['eligible'];reason=case['skip_reason'];views=[];native=[];scores=[];score_reuse=None
            if eligible:
                pairpath=Path(indexed[name]['path']);pins[str(pairpath)]=sha(pairpath)
                if pins[str(pairpath)]!=indexed[name]['sha256']:raise ValueError('paired cache changed')
                pair=load(pairpath);frame=pair['teacher']
                views=[copy.deepcopy(v) for v in case['new_voter_views'] if v['weight_sha256'] in (NEW_SHA,WEIGHT_SHA)]
                if {v['weight_sha256'] for v in views}!={NEW_SHA,WEIGHT_SHA}:raise ValueError('missing reused actual roles')
                image=read_image(source);expected,valid=expected_in_source(reference,alignment['source_to_reference_homography'],image.shape[:2])
                progress('fresh_student_global');pred=full_frame(student,image);fresh+=1
                views.append(dict(source_sha256=pins[str(source)],weight_sha256=STUDENT_SHA,predictions=pred,view='fresh_global960'))
                if 'windows' in pair['student']['predictions']:
                    views.append(cached_view(pair['student'],pins[str(source)],STUDENT_SHA,1280,960))
                else:
                    if pair['student']['predictions']['merged_predictions'] or pair['student']['predictions']['edge_kept_predictions']:
                        raise ValueError('nonempty student cache lacks options')
                    progress('fresh_missing_student_native');pred=predict(student,image);fresh+=1
                    views.append(dict(source_sha256=pins[str(source)],weight_sha256=STUDENT_SHA,predictions=pred,view='fresh_native1280stride960'))
                finepath=FINE/'train'/(stem+'_predictions.json');pins[str(finepath)]=sha(finepath);fine=load(finepath)
                student_fine=[v for v in fine['new_views'] if v['weight_sha256']==STUDENT_SHA]
                if student_fine:
                    if len(student_fine)!=1:raise ValueError('ambiguous fine student cache')
                    views.append(cached_view(student_fine[0],pins[str(source)],STUDENT_SHA,960,720))
                else:
                    progress('fresh_missing_student_fine');pred=fine_infer(student,image);fresh+=1
                    views.append(dict(source_sha256=pins[str(source)],weight_sha256=STUDENT_SHA,predictions=pred,view='fresh_fine960stride720'))
                views=validate_views(views,roles,pins[str(source)],image.shape[:2])
                native=proposals(frame,views,current);native=[native[i] for i in valid_boxes([r['box_xyxy'] for r in native],valid)]
                native=attach(native,views,pins[str(source)],image.shape[:2])
                if native:
                    old_binding=semantic_binding(case['source_sha256'],prior['pins'][str(referencepath)],prior['runtime'],case['alignment'],image.shape[:2])
                    new_binding=semantic_binding(pins[str(source)],REFERENCE_SHA,frozen,alignment,image.shape[:2])
                    scores,missing,reused=semantic_reuse(native,case['proposals'],case['probabilities'],old_binding,new_binding)
                    score_reuse=dict(source_case_path=str(casepath),source_case_sha256=pins[str(casepath)],
                        old_input_binding=old_binding,new_input_binding=new_binding,reused_count=reused,fresh_indices=missing,
                        same_classifier_not_an_extra_vote=True)
                    reused_scores+=reused
                    if missing:
                        progress('fresh_missing_paired_semantics')
                        if encoder is None:
                            import dino_feature_diff as dino
                            encoder=dino._model();encoder.eval().requires_grad_(False);torch.set_num_threads(2)
                        boxes=[native[i]['box_xyxy'] for i in missing]
                        vectors=paired_features(embeddings(encoder,image,boxes),embeddings(encoder,expected,boxes))
                        with torch.inference_mode():new_scores=head(vectors).softmax(1).tolist()
                        for i,value in zip(missing,new_scores):scores[i]=value
                        fresh_scores+=len(missing)
            trial=select(current,native,scores,HEAD_SHA);check_prefix(original,current,trial);candidate_count+=len(native)
            save(folder/(stem+'_predictions.json'),dict(image=name,original=original,current=current,trial=trial,alignment=alignment,eligible=eligible,
                skip_reason=reason,source_sha256=pins[str(source)],head_sha256=HEAD_SHA,roles=roles,new_voter_views=views,proposals=native,probabilities=scores,
                semantic_score_reuse=score_reuse))
            targets=read_targets('train',name,[2736,3648],entry['label_sha256'],pins)
            a=matches(current['all_predictions'],targets)[0];b=matches(trial['all_predictions'],targets)[0];o=matches(original['all_predictions'],targets)[0]
            rows.append(dict(image=name,original=metric(original['all_predictions'],targets),current=metric(current['all_predictions'],targets),trial=metric(trial['all_predictions'],targets),
                gained=sorted(b-a),lost=sorted(a-b),lost_original=sorted(o-b),skip_reason=reason))
            save(OUT/'partial_metrics.json',dict(status='incomplete_not_acceptance',completed=index+1,total=192,cases=rows))
            print(dict(completed=index+1,total=192,image=name,trial_tp=rows[-1]['trial']['tp'],fresh_views=fresh),flush=True)
        totals={v:{k:sum(r[v][k] for r in rows) for k in ('tp','unmatched','fn','predictions','targets')} for v in ('original','current','trial')}
        check_totals(totals)
        normal=sum(r['trial']['predictions'] for r in rows if r['image'].startswith('normal_'))
        passed=totals['trial']['tp']>298 and totals['trial']['unmatched']<=4 and normal==0 and not any(r['lost'] or r['lost_original'] for r in rows)
        if any(sha(p)!=d for p,d in pins.items()) or any(sha(p)!=d for p,d in prior['pins'].items()) or native_pose_runtime_fingerprint(REPO)!=frozen or dirty()!=before_dirty or CONFIG!=original_config:
            raise ValueError('source/mainline/cached evidence changed')
        result=dict(status='source_pass_requires_independent_replay_and_fresh_holdouts' if passed else 'rejected',qualifies=passed,summary=totals,cases=rows,normal_cues=normal,
            pins=pins,runtime=frozen,seconds=round(time.monotonic()-started,2),fresh_view_calls=fresh,proposals=candidate_count,no_heldout_reads=True,no_deployment=True,field_accuracy=None,mainline_unchanged=True)
        result.update(reused_semantic_scores=reused_scores,fresh_semantic_scores=fresh_scores,cache_is_not_an_extra_vote=True)
        save(OUT/'report.json',result);save(OUT/'progress.json',{k:result[k] for k in ('status','seconds','fresh_view_calls','proposals')})
        print(dict(status=result['status'],summary=totals),flush=True)
    except BaseException as exc:
        save(OUT/'progress.json',dict(status='failed',completed=len(rows),total=192,error=type(exc).__name__+': '+str(exc),partial_is_not_accuracy_acceptance=True));raise


if __name__=='__main__':main()
