"""Conditional fresh3-role INNER/OUTER; no fitting, no deployment."""
import copy
import os
import shutil
import sys
import time
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,REPO,DATA,load,save,sha,read_image,REFERENCE_SHA
from current_port_baseline_audit import BASE,read_current_case,read_targets
from run_allport480_source import NEW_SHA,TRAIN,full_frame
from inspection_agent.paired_native_pose import native_pose_runtime_fingerprint,HEAD_SHA,HEAD_RELATIVE
from inspection_agent.paired_port_features import expected_in_source,valid_boxes,embeddings,paired_features
from inspection_agent.optional_port_crop_review import CONFIG,predict
from inspection_agent.teacher_student_port_support import STUDENT_SHA,STUDENT_RELATIVE
from inspection_agent.feature_residual_port_support import WEIGHT_SHA,WEIGHT_RELATIVE
from paired_fine_tile_views import infer as fine_infer
from paired_port_semantics import CachedReferenceSIFT
from replacement_voters import validate_views,check_prefix
from raw_pose_consensus import proposals
from fine_voter_quality import attach
from consensus_rank import select
from audit_port_multiscale_acceptance import metric,matches

SOURCE=ROOT/'artifacts/allport480_teacher_source_20261005'
AUDIT=ROOT/'artifacts/allport480_teacher_source_audit_20261005/report.json'
OUT=ROOT/'artifacts/allport480_teacher_holdouts_20261005'
CURRENT=ROOT/'artifacts/paired_pose_native_three_20261004'
PLAN=ROOT/'artifacts/allport480_teacher_holdouts_preregistration_20261005/PLAN.md'
GROUPS=ROOT/'artifacts/port_training_multiscale_20261002/protocol.json'


def check_stage_membership(stage,names,accepted_names,train_names,inner_names):
    if stage not in ('inner','outer'):raise ValueError('unknown fixed validation stage')
    count=48 if stage=='inner' else 30
    if len(names)!=count or len(set(names))!=count or len(accepted_names)!=count or set(names)!=set(accepted_names):
        raise ValueError('validation membership differs from accepted stage')
    if any(Path(name).name!=name for name in names):raise ValueError('invalid validation source name')
    if stage=='inner' and (set(names)!=set(inner_names) or set(names)&set(train_names)):
        raise ValueError('INNER split drift or source overlap')
    # OUTER is a different physical directory val01. Equal basenames in train01
    # are not identity; source SHA disjointness is checked when pixels are bound.
    return True


def stage_qualifies(stage,totals,normal,rows):
    if stage not in ('inner','outer'):raise ValueError('unknown fixed validation stage')
    tp=totals['trial']['tp'];unmatched=totals['trial']['unmatched']
    gain=tp>68 if stage=='inner' else tp>=40
    return gain and unmatched<=(0 if stage=='inner' else 1) and normal==0 and not any(r['lost'] or r['lost_original'] for r in rows)


def main():
    if OUT.exists():raise FileExistsError('preserve frozen development validation')
    # Preconditions precede ANY validation index/label/image read.
    source=load(SOURCE/'report.json');audit=load(AUDIT)
    if (source['status']!='source_pass_requires_independent_replay_and_fresh_holdouts' or not source['qualifies']
        or audit['status']!='pass' or not audit['candidate_source_qualifies']
        or audit['source_report_sha256']!=sha(SOURCE/'report.json')
        or len(audit['source_case_sha256'])!=192 or any(sha(p)!=v for p,v in audit['source_case_sha256'].items())
        or any(sha(p)!=v for p,v in source['pins'].items())):raise ValueError('complete independent strict source pass required')
    frozen=native_pose_runtime_fingerprint(REPO)
    if frozen!=source['runtime']:raise ValueError('mainline drift')
    original_config=copy.deepcopy(CONFIG);roles=dict(teacher=NEW_SHA,student=STUDENT_SHA,feature=WEIGHT_SHA)
    if load(SOURCE/'protocol.json')['roles']!=roles:raise ValueError('fixed source role population changed')
    training=load(TRAIN/'full/report.json');paths=dict(teacher=Path(training['last_checkpoint']),student=REPO/STUDENT_RELATIVE,feature=REPO/WEIGHT_RELATIVE)
    if any(sha(p)!=roles[k] for k,p in paths.items()) or sha(REPO/HEAD_RELATIVE)!=HEAD_SHA:raise ValueError('fixed weight identity changed')
    referencepath=DATA/'images/train01/normal_073.JPG'
    if sha(referencepath)!=REFERENCE_SHA:raise ValueError('reference changed')
    code=[Path(__file__),PLAN,SOURCE/'report.json',SOURCE/'protocol.json',AUDIT,TRAIN/'full/report.json',
        CURRENT/'report.json',GROUPS,referencepath,REPO/HEAD_RELATIVE,*paths.values()]
    code += [Path(__file__).with_name(n+'.py') for n in ('run_allport480_source','replacement_voters','raw_pose_consensus',
        'fine_voter_quality','consensus_rank','novel_box_geometry','relative_port_box','paired_fine_tile_views',
        'prepare_paired_port_semantics','current_port_baseline_audit','audit_port_multiscale_acceptance','paired_port_semantics')]
    pins={str(p):sha(p) for p in code};dirty=lambda:__import__('subprocess').check_output(['E:/Git/cmd/git.exe','-C',str(REPO),'status','--porcelain'],text=True)
    groups=load(GROUPS)
    train_sha={value for path,value in source['pins'].items() if Path(path).parent==DATA/'images/train01'}
    if len(groups['train_sources'])!=192 or len(train_sha)!=192:raise ValueError('complete bound source population required')
    before_dirty=dirty()
    import psutil,cv2,torch
    if psutil.virtual_memory().available<6*2**30:raise RuntimeError('protect active inference memory')
    torch.set_num_threads(2);cv2.setNumThreads(1)
    OUT.mkdir();(OUT/'config/Ultralytics').mkdir(parents=True);shutil.copy2('C:/Windows/Fonts/arial.ttf',OUT/'config/Ultralytics/Arial.ttf')
    os.environ.update(YOLO_CONFIG_DIR=str(OUT/'config'),YOLO_OFFLINE='True',YOLO_AUTOINSTALL='False',HF_HUB_OFFLINE='1')
    from ultralytics import YOLO
    models={}
    for role,path in paths.items():
        model=YOLO(str(path))
        if model.task!='segment' or dict(model.names)!={0:'unplugged_plug',1:'unplugged_jack'}:raise ValueError('detector class contract changed')
        class Capped:
            def __init__(self,m):self.model=m
            def predict(self,*a,**kw):
                torch.set_num_threads(2);result=self.model.predict(*a,**kw);torch.set_num_threads(2);return result
        models[role]=Capped(model)
    head=torch.nn.Linear(6144,3);head.load_state_dict(torch.load(REPO/HEAD_RELATIVE,map_location='cpu',weights_only=True)['state_dict']);head.eval().requires_grad_(False)
    import assembly_auto_review_robust_v3 as registration
    reference=read_image(referencepath);encoder=None;stages={};started=time.monotonic();fresh_views=0;fresh_scores=0
    save(OUT/'protocol.json',dict(pins=pins,runtime=frozen,roles=roles,original_config=original_config,mainline_dirty_before=before_dirty,
        no_fitting=True,no_deployment=True,field_accuracy=None,reused_development_groups=True,fresh_semantic_scores_only=True,
        same_weight_views_one_vote=True,GT_after_predictions=True,wall_bound_seconds=7200,
        stage_gates=dict(inner=dict(tp_strictly_above=68,unmatched_max=0,count=48),outer=dict(tp_at_least=40,unmatched_max=1,count=30))))
    def finish(failed):
        if (any(sha(p)!=d for p,d in pins.items()) or native_pose_runtime_fingerprint(REPO)!=frozen
            or CONFIG!=original_config or dirty()!=before_dirty):raise ValueError('validation input/runtime/dirty state drift')
        result=dict(status='rejected' if failed else 'development_pass_requires_independent_actual_reference_ROI_Qt_SAM',failed_stage=failed,
            stages=stages,pins=pins,runtime=frozen,seconds=round(time.monotonic()-started,2),fresh_view_calls=fresh_views,
            fresh_semantic_scores=fresh_scores,no_fitting=True,no_deployment=True,field_accuracy=None,
            reused_development_groups=True,mainline_unchanged=True)
        save(OUT/'report.json',result);save(OUT/'progress.json',dict(status=result['status'],failed_stage=failed,seconds=result['seconds']))
        print(dict(status=result['status'],stages=stages),flush=True)
    try:
        with CachedReferenceSIFT(reference):
            for stage,count,base_tp,base_fp in [('inner',48,68,0),('outer',30,40,1)]:
                stagepath=BASE/stage/'report.json';pins[str(stagepath)]=sha(stagepath);entries=load(stagepath)['cases']
                accepted_stagepath=CURRENT/stage/'report.json';pins[str(accepted_stagepath)]=sha(accepted_stagepath)
                accepted_stage=load(accepted_stagepath)
                check_stage_membership(stage,[r['image'] for r in entries],[r['image'] for r in accepted_stage['cases']],groups['train_sources'],groups['inner_val_sources'])
                folder=OUT/stage;folder.mkdir();rows=[]
                for index,entry in enumerate(entries):
                    if time.monotonic()-started>7200:raise TimeoutError('partial validation is not acceptance')
                    name=entry['image'];stem=Path(name).stem
                    progress=lambda phase:save(OUT/'progress.json',dict(status='running',pid=os.getpid(),stage=stage,completed=index,total=count,
                        image=name,phase=phase,fresh_view_calls=fresh_views,fresh_semantic_scores=fresh_scores,seconds=round(time.monotonic()-started,2)))
                    progress('bound_original_source')
                    teacher,original=read_current_case(stage,entry,pins)
                    acceptedpath=CURRENT/stage/(stem+'_predictions.json');pins[str(acceptedpath)]=sha(acceptedpath);accepted=load(acceptedpath)
                    if accepted['head_sha256']!=HEAD_SHA:raise ValueError('accepted prefix head changed')
                    current=accepted['trial'];check_prefix(original['trial'],current,current)
                    sourcepath=DATA/'images'/('val01' if stage=='outer' else 'train01')/name;pins[str(sourcepath)]=sha(sourcepath)
                    if pins[str(sourcepath)] in train_sha:raise ValueError('validation pixels duplicate training source')
                    remaining=5-(len(current['all_predictions'])-len(current['primary']))
                    views=[];native=[];scores=[];alignment=None;eligible=False;reason='shared_budget_full'
                    if remaining>0:
                        image=read_image(sourcepath)
                        if image.shape[:2]!=(2736,3648):raise ValueError('validation source frame changed')
                        progress('fresh_original_SIFT');cv2.setRNGSeed(0);_,alignment=registration.automatic_homography(reference,image)
                        if alignment.get('alignment_quality',{}).get('reliable') is True:
                            eligible=True;reason=None;expected,valid=expected_in_source(reference,alignment['source_to_reference_homography'],image.shape[:2])
                            for role in roles:
                                for viewname,infer in [('fresh_global960',full_frame),('fresh_native1280stride960',predict),('fresh_fine960stride720',fine_infer)]:
                                    progress(role+'_'+viewname);prediction=infer(models[role],image);fresh_views+=1
                                    views.append(dict(source_sha256=pins[str(sourcepath)],weight_sha256=roles[role],view=viewname,predictions=prediction))
                            views=validate_views(views,roles,pins[str(sourcepath)],image.shape[:2])
                            native=proposals(teacher,views,current);native=[native[i] for i in valid_boxes([p['box_xyxy'] for p in native],valid)]
                            native=attach(native,views,pins[str(sourcepath)],image.shape[:2])
                            if native:
                                progress('fresh_frozen_paired_semantics')
                                if encoder is None:
                                    import dino_feature_diff as dino
                                    encoder=dino._model();encoder.eval().requires_grad_(False);torch.set_num_threads(2)
                                boxes=[p['box_xyxy'] for p in native];vectors=paired_features(embeddings(encoder,image,boxes),embeddings(encoder,expected,boxes))
                                with torch.inference_mode():scores=head(vectors).softmax(1).tolist()
                                fresh_scores+=len(native)
                        else:reason='unreliable_fresh_reference_alignment'
                    trial=select(current,native,scores,HEAD_SHA);check_prefix(original['trial'],current,trial)
                    save(folder/(stem+'_predictions.json'),dict(image=name,original=original['trial'],current=current,trial=trial,source_sha256=pins[str(sourcepath)],
                        head_sha256=HEAD_SHA,roles=roles,alignment=alignment,eligible=eligible,skip_reason=reason,new_voter_views=views,proposals=native,probabilities=scores))
                    targets=read_targets(stage,name,[2736,3648],entry['label_sha256'],pins)
                    a=matches(current['all_predictions'],targets)[0];b=matches(trial['all_predictions'],targets)[0];o=matches(original['trial']['all_predictions'],targets)[0]
                    rows.append(dict(image=name,original=metric(original['trial']['all_predictions'],targets),current=metric(current['all_predictions'],targets),
                        trial=metric(trial['all_predictions'],targets),gained=sorted(b-a),lost=sorted(a-b),lost_original=sorted(o-b),skip_reason=reason))
                    save(folder/'partial_metrics.json',dict(status='incomplete_not_acceptance',completed=index+1,total=count,cases=rows))
                    print(dict(stage=stage,completed=index+1,total=count,image=name),flush=True)
                totals={v:{k:sum(r[v][k] for r in rows) for k in ('tp','unmatched','fn','predictions','targets')} for v in ('original','current','trial')}
                if totals['current']['tp']!=base_tp or totals['current']['unmatched']!=base_fp:raise ValueError('accepted validation baseline changed')
                normal=sum(r['trial']['predictions'] for r in rows if r['image'].startswith('normal_'))
                qualifies=stage_qualifies(stage,totals,normal,rows)
                stages[stage]=dict(qualifies=qualifies,summary=totals,normal_cues=normal,count=count)
                save(folder/'report.json',dict(status='complete',**stages[stage],cases=rows))
                if not qualifies:finish(stage);return
        finish(None)
    except BaseException as exc:
        save(OUT/'progress.json',dict(status='failed',error=type(exc).__name__+': '+str(exc),partial_is_not_acceptance=True));raise


if __name__=='__main__':main()
