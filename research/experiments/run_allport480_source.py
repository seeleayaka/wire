"""Actual TRAIN192 inference of frozen last.pt; replacement, not fourth vote."""
import copy
import os
import shutil
import sys
import time
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,REPO,DATA,load,save,sha,read_image,REFERENCE_SHA
from current_port_baseline_audit import BASE,read_current_case,read_targets
from inspection_agent.paired_native_pose import native_pose_runtime_fingerprint,HEAD_SHA,HEAD_RELATIVE
from inspection_agent.paired_port_features import expected_in_source,valid_boxes,embeddings,paired_features
from inspection_agent.optional_port_crop_review import CONFIG,predict
from inspection_agent.teacher_student_port_support import TEACHER_SHA,TEACHER_RELATIVE
from inspection_agent.feature_residual_port_support import WEIGHT_SHA,WEIGHT_RELATIVE
from inspection_agent.port_tiling import tile_windows,merge_tiled_ports
from paired_fine_tile_views import infer as fine_infer
from raw_pose_consensus import proposals
from fine_voter_quality import attach
from consensus_rank import select
from replacement_voters import validate_views,check_prefix
from audit_port_multiscale_acceptance import metric,matches

OUT=ROOT/'artifacts/allport480_source_20261005'
TRAIN=ROOT/'artifacts/allport480_training_20261004'
PLAN=ROOT/'artifacts/allport480_source_preregistration_20261005/PLAN.md'
PRIOR=ROOT/'artifacts/allport480_source_preregistration_20261004/PLAN.md'
RESEARCH=ROOT/'artifacts/fine_consensus_rank_20261004'
FINE=ROOT/'artifacts/paired_fine_tile_views_20261003/full'
PAIR=ROOT/'artifacts/paired_support_graph_20261003/source_selections/train/index.json'
NEW_SHA='7a893f494eb4e2433712047abcd58fb1e036043da96e31bca6bd0e77bed143ec'


def cached_view(view,source_sha,weight_sha,tile_size,stride):
    if view['source_sha256']!=source_sha or view['weight_sha256']!=weight_sha:
        raise ValueError('cached frame/weight mismatch')
    pred=view['predictions']
    if pred['source_shape']!=[2736,3648] or pred.get('windows')!=[list(w) for w in tile_windows(3648,2736,tile_size,stride)]:
        raise ValueError('cached geometry/options mismatch')
    if pred['merged_predictions']!=merge_tiled_ports(pred['edge_kept_predictions'],CONFIG['cross_tile_nms_iou']):
        raise ValueError('cached merge replay mismatch')
    return copy.deepcopy(view)


def full_frame(model,image):
    output=model.predict(image,imgsz=CONFIG['predict_imgsz'],conf=CONFIG['predict_conf_floor'],
                         iou=CONFIG['predict_iou'],max_det=CONFIG['predict_max_det'],device='cpu',verbose=False,save=False)
    if len(output)!=1:raise ValueError('full frame count mismatch')
    rows=[]
    for box in output[0].boxes:
        rows.append(dict(box_xyxy=list(map(float,box.xyxy[0].tolist())),confidence=float(box.conf.item()),
                         class_id=int(box.cls.item()),support_tiles=[0],source_tile=0))
    return dict(source_shape=list(image.shape[:2]),windows=[[0,0,image.shape[1],image.shape[0]]],merged_predictions=rows)


def main():
    if OUT.exists():raise FileExistsError('preserve existing replacement experiment')
    import psutil,cv2,torch
    if psutil.virtual_memory().available<6*2**30:raise RuntimeError('insufficient available memory; do not compete with active jobs')
    torch.set_num_threads(2);cv2.setNumThreads(1)
    training=load(TRAIN/'full/report.json');audit=load(TRAIN/'dataset_audit/report.json')
    manifest=TRAIN/'dataset/dataset_manifest.json';dataset=load(manifest)
    if not (training['status']=='complete' and training['completed_epochs']==2 and training['finite_gradients'] and
            training['nonzero_gradient_steps']>0 and training['changed_trainable_parameters']>0 and
            training['final_checkpoint_policy']=='last.pt' and training['last_sha256']==NEW_SHA):
        raise ValueError('training prerequisite failed')
    if training['dataset_manifest_sha256']!=sha(manifest) or training['plan_sha256']!=sha(TRAIN/'PLAN.md') or training['script_sha256']!=sha(ROOT/'experiments/train_allport480.py'):
        raise ValueError('training provenance drift')
    if audit['status']!='complete' or audit['manifest_sha256']!=sha(manifest) or audit['counts']!=dict(old_rows_byte_identical=656,new_JPEG_byte_reproductions=319,new_source_label_replays=319,new_label_instances=560):
        raise ValueError('independent data audit failed')
    if audit['auditor_sha256']!=sha(ROOT/'experiments/audit_allport480_dataset.py'):
        raise ValueError('dataset audit code drift')
    if any(sha(p)!=d for p,d in dataset['protected_pins'].items()):raise ValueError('protected dataset source drift')
    if any(sha(TRAIN/'dataset'/r[k])!=r[k+'_sha256'] for r in dataset['records'] for k in ('image','label')):
        raise ValueError('training bytes changed')
    frozen=native_pose_runtime_fingerprint(REPO);prior=load(RESEARCH/'report.json')
    if prior['runtime']!=frozen or not prior['stages']['train']['qualifies'] or any(sha(p)!=d for p,d in prior['pins'].items()):
        raise ValueError('research prefix provenance drift')
    fine_protocol=load(FINE/'protocol.json');fine_report=load(FINE/'report.json')
    if fine_protocol['original_config']!=CONFIG or any(sha(p)!=d for p,d in fine_protocol['pins'].items()):
        raise ValueError('fine cached inference options/code drift')
    newweight=Path(training['last_checkpoint'])
    weights=dict(teacher=REPO/TEACHER_RELATIVE,student=newweight,feature=REPO/WEIGHT_RELATIVE)
    roles=dict(teacher=TEACHER_SHA,student=NEW_SHA,feature=WEIGHT_SHA)
    if any(sha(weights[k])!=roles[k] for k in roles):raise ValueError('detector weight drift')
    if sha(REPO/HEAD_RELATIVE)!=HEAD_SHA:raise ValueError('paired head drift')
    entries=load(BASE/'train/report.json')['cases'];indexed={r['image']:r for r in load(PAIR)['records']}
    names=load(ROOT/'artifacts/port_training_multiscale_20261002/protocol.json')['train_sources']
    if len(entries)!=192 or sorted(r['image'] for r in entries)!=sorted(names):raise ValueError('incomplete TRAIN membership')
    code=[Path(__file__),PLAN,PRIOR,TRAIN/'full/report.json',TRAIN/'dataset_audit/report.json',manifest,TRAIN/'PLAN.md',
          ROOT/'experiments/train_allport480.py',ROOT/'experiments/audit_allport480_dataset.py',PAIR,
          FINE/'protocol.json',FINE/'report.json',RESEARCH/'report.json',REPO/HEAD_RELATIVE]
    code += [Path(__file__).with_name(n+'.py') for n in ('replacement_voters','raw_pose_consensus','fine_voter_quality','consensus_rank','novel_box_geometry','relative_port_box','paired_fine_tile_views','current_port_baseline_audit','prepare_paired_port_semantics','audit_port_multiscale_acceptance')]
    pins={str(p):sha(p) for p in code+list(weights.values())};referencepath=DATA/'images/train01/normal_073.JPG'
    if sha(referencepath)!=REFERENCE_SHA:raise ValueError('reference drift')
    pins[str(referencepath)]=sha(referencepath)
    dirty=lambda:__import__('subprocess').check_output(['E:/Git/cmd/git.exe','-C',str(REPO),'status','--porcelain'],text=True)
    before_dirty=dirty();original_config=copy.deepcopy(CONFIG)
    OUT.mkdir();(OUT/'config/Ultralytics').mkdir(parents=True)
    shutil.copy2('C:/Windows/Fonts/arial.ttf',OUT/'config/Ultralytics/Arial.ttf')
    os.environ.update(YOLO_CONFIG_DIR=str(OUT/'config'),YOLO_OFFLINE='True',YOLO_AUTOINSTALL='False',HF_HUB_OFFLINE='1')
    from ultralytics import YOLO
    class Capped:
        def __init__(self,model):self.model=model
        def predict(self,*a,**kw):
            torch.set_num_threads(2);output=self.model.predict(*a,**kw);torch.set_num_threads(2);return output
    models={}
    for role,path in weights.items():
        model=YOLO(str(path))
        if model.task!='segment' or dict(model.names)!={0:'unplugged_plug',1:'unplugged_jack'}:raise ValueError('detector class contract mismatch')
        models[role]=Capped(model)
    head=torch.nn.Linear(6144,3);head.load_state_dict(torch.load(REPO/HEAD_RELATIVE,map_location='cpu',weights_only=True)['state_dict'])
    head.eval().requires_grad_(False);encoder=None;reference=read_image(referencepath)
    started=time.monotonic();rows=[];fresh_calls=0;candidate_count=0;folder=OUT/'train';folder.mkdir()
    save(OUT/'protocol.json',dict(pins=pins,runtime=frozen,mainline_dirty_before=before_dirty,roles=roles,
         original_config=original_config,train_sources=names,wall_bound_seconds=7200,plan_sha256=sha(PLAN),
         head_sha256=HEAD_SHA,protected_prefixes=[295,298],no_heldout_reads=True,no_deployment=True,
         views=['global960','native1280stride960','fine960stride720'],same_weight_one_vote=True,GT_after_predictions=True))
    try:
        for index,entry in enumerate(entries):
            if time.monotonic()-started>7200:raise TimeoutError('operational wall bound; partial source is not accuracy acceptance')
            name=entry['image'];stem=Path(name).stem
            progress=lambda phase:save(OUT/'progress.json',dict(status='running',pid=os.getpid(),completed=index,total=192,
                image=name,phase=phase,seconds=round(time.monotonic()-started,2),fresh_view_calls=fresh_calls,proposals=candidate_count))
            progress('prepare_verified_source')
            source=DATA/'images/train01'/name;pins[str(source)]=sha(source)
            pairpath=Path(indexed[name]['path'])
            if sha(pairpath)!=indexed[name]['sha256']:raise ValueError('paired detector cache drift')
            pins[str(pairpath)]=sha(pairpath);paired=load(pairpath)
            teacher,original=read_current_case('train',entry,pins)
            if teacher!=paired['teacher']:raise ValueError('teacher cache mismatch')
            currentpath=RESEARCH/'train'/(stem+'_predictions.json');pins[str(currentpath)]=sha(currentpath)
            current=load(currentpath)['trial'];check_prefix(original['trial'],current,current)
            alignmentpath=FINE/'train'/(stem+'_predictions.json');pins[str(alignmentpath)]=sha(alignmentpath)
            cache=load(alignmentpath);alignment=cache['alignment']
            if alignment is not None and fine_report['pins'].get(str(source))!=pins[str(source)]:
                raise ValueError('cached alignment source pin mismatch')
            # Some no-room cases carry no alignment; they must not acquire cues.
            remaining=5-(len(current['all_predictions'])-len(current['primary']))
            eligible=remaining>0 and alignment is not None and alignment.get('alignment_quality',{}).get('reliable') is True
            views=[];native=[];scores=[];reason=None
            if eligible:
                image=read_image(source)
                if list(image.shape[:2])!=[2736,3648]:raise ValueError('source geometry drift')
                expected,valid=expected_in_source(reference,alignment['source_to_reference_homography'],image.shape[:2])
                for role in ('teacher','feature'):
                    if 'windows' in paired[role]['predictions']:
                        views.append(cached_view(paired[role],pins[str(source)],roles[role],1280,960))
                    else:
                        # Old short-circuit empty placeholders are NOT actual negative inference.
                        if paired[role]['predictions']['merged_predictions'] or paired[role]['predictions']['edge_kept_predictions']:
                            raise ValueError('nonempty native cache lacks inference provenance')
                        progress('fresh_missing_native_'+role);raw=predict(models[role],image);fresh_calls+=1
                        views.append(dict(source_sha256=pins[str(source)],weight_sha256=roles[role],predictions=raw,view='fresh_missing_native1280stride960'))
                teacher_fine=[v for v in cache['new_views'] if v['weight_sha256']==TEACHER_SHA]
                for role in ('teacher','student','feature'):
                    progress('fresh_global_'+role)
                    raw=full_frame(models[role],image);fresh_calls+=1
                    views.append(dict(source_sha256=pins[str(source)],weight_sha256=roles[role],predictions=raw,view='fresh_global960'))
                    if role=='student':
                        progress('fresh_native_student');raw=predict(models[role],image);fresh_calls+=1
                        views.append(dict(source_sha256=pins[str(source)],weight_sha256=roles[role],predictions=raw,view='fresh_native1280stride960'))
                    if role=='teacher' and teacher_fine:
                        if len(teacher_fine)!=1:raise ValueError('ambiguous fine teacher cache')
                        if sha(source)!=fine_report['pins'][str(source)]:raise ValueError('fine source pin drift')
                        views.append(cached_view(teacher_fine[0],pins[str(source)],roles[role],960,720))
                    else:
                        progress('fresh_fine_'+role);raw=fine_infer(models[role],image);fresh_calls+=1
                        views.append(dict(source_sha256=pins[str(source)],weight_sha256=roles[role],predictions=raw,view='fresh_fine960stride720'))
                views=validate_views(views,roles,pins[str(source)],image.shape[:2])
                native=proposals(teacher,views,current)
                native=[native[i] for i in valid_boxes([p['box_xyxy'] for p in native],valid)]
                native=attach(native,views,pins[str(source)],image.shape[:2])
                if native:
                    progress('frozen_paired_semantics')
                    if encoder is None:
                        import dino_feature_diff as dino
                        encoder=dino._model();encoder.eval().requires_grad_(False);torch.set_num_threads(2)
                    boxes=[p['box_xyxy'] for p in native]
                    vectors=paired_features(embeddings(encoder,image,boxes),embeddings(encoder,expected,boxes))
                    with torch.inference_mode():scores=head(vectors).softmax(1).tolist()
            else:reason='shared_budget_full' if remaining==0 else 'unreliable_or_unavailable_reference_alignment'
            trial=select(current,native,scores,HEAD_SHA);check_prefix(original['trial'],current,trial)
            candidate_count+=len(native)
            save(folder/(stem+'_predictions.json'),dict(image=name,original=original['trial'],current=current,trial=trial,
                 source_sha256=pins[str(source)],head_sha256=HEAD_SHA,proposals=native,probabilities=scores,
                 new_voter_views=views,alignment=alignment,eligible=eligible,skip_reason=reason,roles=roles))
            targets=read_targets('train',name,[2736,3648],entry['label_sha256'],pins)
            a=matches(current['all_predictions'],targets)[0];b=matches(trial['all_predictions'],targets)[0]
            o=matches(original['trial']['all_predictions'],targets)[0]
            rows.append(dict(image=name,original=metric(original['trial']['all_predictions'],targets),current=metric(current['all_predictions'],targets),
                 trial=metric(trial['all_predictions'],targets),gained=sorted(b-a),lost=sorted(a-b),lost_original=sorted(o-b),skip_reason=reason))
            print(dict(completed=index+1,total=192,image=name,current=rows[-1]['current']['tp'],trial=rows[-1]['trial']['tp'],fresh_views=fresh_calls),flush=True)
            save(OUT/'partial_metrics.json',dict(status='incomplete_not_acceptance',completed=index+1,total=192,cases=rows))
        totals={v:{k:sum(r[v][k] for r in rows) for k in ('tp','unmatched','fn','predictions','targets')} for v in ('original','current','trial')}
        if (totals['original']['tp'],totals['original']['unmatched'],totals['current']['tp'],totals['current']['unmatched'])!=(295,4,298,4):
            raise ValueError('protected baseline metrics mismatch')
        normal=sum(r['trial']['predictions'] for r in rows if r['image'].startswith('normal_'))
        passes=totals['trial']['tp']>298 and totals['trial']['unmatched']<=4 and normal==0 and not any(r['lost'] or r['lost_original'] for r in rows)
        if any(sha(p)!=d for p,d in pins.items()) or native_pose_runtime_fingerprint(REPO)!=frozen or CONFIG!=original_config or dirty()!=before_dirty:
            raise ValueError('source/code/model/mainline drift')
        result=dict(status='source_pass_requires_independent_replay_and_fresh_holdouts' if passes else 'rejected',
            summary=totals,cases=rows,normal_cues=normal,qualifies=passes,pins=pins,runtime=frozen,
            seconds=round(time.monotonic()-started,2),fresh_view_calls=fresh_calls,proposals=candidate_count,
            no_heldout_reads=True,no_deployment=True,field_accuracy=None,mainline_unchanged=True)
        save(OUT/'report.json',result);save(OUT/'progress.json',{k:result[k] for k in ('status','seconds','fresh_view_calls','proposals')})
        print(dict(status=result['status'],summary=totals,normal_cues=normal),flush=True)
    except BaseException as error:
        save(OUT/'progress.json',dict(status='failed',pid=os.getpid(),completed=len(rows),total=192,seconds=round(time.monotonic()-started,2),
            error=type(error).__name__+': '+str(error),partial_is_not_accuracy_acceptance=True));raise


if __name__=='__main__':main()
