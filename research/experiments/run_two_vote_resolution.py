"""TRAIN-only missing-checkpoint ROI evidence, fixed semantic and consensus gates."""
import copy
import os
import shutil
import statistics
import sys
import time
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,REPO,DATA,load,save,sha,read_image,REFERENCE_SHA
from current_port_baseline_audit import BASE,read_targets
from inspection_agent.paired_native_pose import native_pose_runtime_fingerprint,HEAD_SHA,HEAD_RELATIVE
from inspection_agent.paired_port_features import expected_in_source,valid_boxes,embeddings,paired_features
from inspection_agent.teacher_student_port_support import TEACHER_RELATIVE,TEACHER_SHA,STUDENT_RELATIVE,STUDENT_SHA
from inspection_agent.feature_residual_port_support import WEIGHT_RELATIVE as FEATURE_RELATIVE,WEIGHT_SHA as FEATURE_SHA
from inspection_agent.optional_port_crop_review import CONFIG
from inspection_agent.port_tiling import near_artificial_edge
from unresolved_voter_seeds import windows
from raw_pose_consensus import proposals
from fine_voter_quality import attach
from consensus_rank import select
from novel_box_geometry import is_duplicate
from audit_port_multiscale_acceptance import metric,matches
PREP=ROOT/'artifacts/two_vote_resolution_preparation_20261004'
OUT=ROOT/'artifacts/two_vote_resolution_20261004'
PLAN=ROOT/'artifacts/two_vote_resolution_preregistration_20261004/PLAN.md'
PREVIOUS=ROOT/'artifacts/fine_consensus_rank_holdouts_20261004/report.json'


def main():
    import cv2
    import torch
    import psutil
    cv2.setNumThreads(1);torch.set_num_threads(2)
    if OUT.exists():raise FileExistsError('Preserve finite missing-checkpoint test')
    assert psutil.virtual_memory().available>6*2**30,'Do not compete for inference memory'
    previous=load(PREVIOUS)
    assert previous['status']=='rejected','Prioritize actual acceptance if previous candidate passed'
    prepared=load(PREP/'report.json');assert prepared['status']=='complete' and prepared['sources']==192
    assert all(sha(Path(p))==v for p,v in prepared['pins'].items())
    frozen=native_pose_runtime_fingerprint(REPO);assert frozen==prepared['runtime']==previous['runtime']
    referencepath=DATA/'images/train01/normal_073.JPG';assert sha(referencepath)==REFERENCE_SHA
    pins={str(p):sha(p) for p in (Path(__file__),PLAN,PREVIOUS,PREP/'report.json',referencepath,REPO/HEAD_RELATIVE,
        Path(__file__).with_name('unresolved_voter_seeds.py'),Path(__file__).with_name('raw_pose_consensus.py'),
        Path(__file__).with_name('fine_voter_quality.py'),Path(__file__).with_name('consensus_rank.py'),
        Path(__file__).with_name('novel_box_geometry.py'),Path(__file__).with_name('relative_port_box.py'),
        REPO/'inspection_agent/paired_port_features.py',REPO/'inspection_agent/paired_native_pose_features.py')}
    assert pins[str(REPO/HEAD_RELATIVE)]==HEAD_SHA
    weights={TEACHER_SHA:REPO/TEACHER_RELATIVE,STUDENT_SHA:REPO/STUDENT_RELATIVE,FEATURE_SHA:REPO/FEATURE_RELATIVE}
    assert len(weights)==3 and all(sha(p)==s for s,p in weights.items());pins.update({str(p):s for s,p in weights.items()})
    OUT.mkdir();configdir=OUT/'config';(configdir/'Ultralytics').mkdir(parents=True)
    shutil.copy2('C:/Windows/Fonts/arial.ttf',configdir/'Ultralytics/Arial.ttf')
    os.environ.update(YOLO_CONFIG_DIR=str(configdir),YOLO_OFFLINE='True',YOLO_AUTOINSTALL='False',HF_HUB_OFFLINE='1')
    from ultralytics import YOLO
    models={};encoder=None;head=torch.nn.Linear(6144,3)
    head.load_state_dict(torch.load(REPO/HEAD_RELATIVE,map_location='cpu',weights_only=True)['state_dict']);head.eval().requires_grad_(False)
    reference=read_image(referencepath);started=time.monotonic();rows=[];folder=OUT/'train';folder.mkdir();original_config=copy.deepcopy(CONFIG)
    entrymap={r['image']:r for r in load(BASE/'train/report.json')['cases']}
    save(OUT/'protocol.json',dict(pins=pins,runtime=frozen,seed_probability_gate=.98,seed_votes=2,final_votes=3,
        local_window=960,offsets=[[-120,-120],[120,120]],input_size=960,head_sha256=HEAD_SHA,
        research_prefix_298_not_installed=True,no_training=True,no_validation_read=True,no_deployment=True,field_accuracy=False))
    progress=lambda **kw:save(OUT/'progress.json',dict(status='running',pid=os.getpid(),seconds=round(time.monotonic()-started,2),**kw))
    total_seed_pairs=0;total_roi_seeds=0;inferred_sources=0;final_proposals=0
    try:
        for completed,record in enumerate(prepared['cases']):
            name=record['image'];progress(phase='semantic_prescreen_then_missing_checkpoint_ROI',image=name,completed=completed,total=192)
            path=Path(record['path']);assert sha(path)==record['sha256'];pins[str(path)]=sha(path);case=load(path)
            sourcepath=DATA/'images/train01'/name;pins[str(sourcepath)]=sha(sourcepath);assert pins[str(sourcepath)]==case['source_sha256']==record['source_sha256']
            current=case['current'];seeds=case['seeds'];total_seed_pairs+=len(seeds);seed_scores=[];chosen=[];roi_evidence=[];extra_models=[];native=[];final_scores=[]
            if seeds:
                import dino_feature_diff as dino
                if encoder is None:encoder=dino._model();encoder.eval().requires_grad_(False);torch.set_num_threads(2)
                image=read_image(sourcepath);expected,mask=expected_in_source(reference,case['alignment']['source_to_reference_homography'],image.shape[:2])
                boxes=[p['box_xyxy'] for p in seeds];assert valid_boxes(boxes,mask)==list(range(len(boxes)))
                vectors=paired_features(embeddings(encoder,image,boxes),embeddings(encoder,expected,boxes))
                with torch.inference_mode():seed_scores=head(vectors).softmax(1).tolist()
                best={}
                for seed,prob in zip(seeds,seed_scores):
                    cls=max(range(3),key=lambda i:prob[i])
                    if cls!=seed['class_id']+1 or prob[cls]<.98:continue
                    quality=float(statistics.median(seed['localization_voter_best_IoU'].values()))
                    rank=(-quality,-prob[cls],*seed['box_xyxy'],seed['class_id']);parent=seed['pose_parent_seed_id']
                    if parent not in best or rank<best[parent][0]:best[parent]=(rank,seed,prob)
                for rank,seed,prob in sorted(best.values(),key=lambda r:r[0]):
                    if len(chosen)>=case['remaining_budget']:break
                    if is_duplicate(seed['box_xyxy'],chosen):continue
                    item=copy.deepcopy(seed);item['prescreen_probability']=prob[seed['class_id']+1];chosen.append(item)
                for seed in chosen:
                    missing=set(weights)-set(seed['semantic_model_vote_sha256']);assert len(missing)==1
                    weight=next(iter(missing));assert len(seed['semantic_model_vote_sha256'])==2
                    if weight not in models:
                        models[weight]=YOLO(str(weights[weight]));assert models[weight].task=='segment' and dict(models[weight].names)=={0:'unplugged_plug',1:'unplugged_jack'}
                    ws=windows(seed,image.shape[:2]);crops=[image[t:b,l:r] for l,t,r,b in ws]
                    torch.set_num_threads(2)
                    outputs=models[weight].predict(crops,imgsz=960,conf=.001,iou=.7,max_det=300,device='cpu',verbose=False,save=False)
                    torch.set_num_threads(2);assert len(outputs)==2
                    views=[];roi_rows=[]
                    for output,window in zip(outputs,ws):
                        l,t,_,_=window;local_rows=[]
                        for box in output.boxes:
                            coords=list(map(float,box.xyxy[0].tolist()))
                            if near_artificial_edge(coords,window,image.shape[1],image.shape[0],16):continue
                            x,y,r,b=coords;row=dict(box_xyxy=[x+l,y+t,r+l,b+t],confidence=float(box.conf.item()),class_id=int(box.cls.item()))
                            local_rows.append(row);roi_rows.append(row)
                        views.append(local_rows)
                    roi_evidence.append(dict(seed=seed,missing_weight_sha256=weight,windows=ws,views=views))
                    extra_models.append(dict(image=name,source_sha256=case['source_sha256'],weight_sha256=weight,
                        predictions=dict(source_shape=list(image.shape[:2]),merged_predictions=roi_rows),view='missing_checkpoint_two_local960_views'))
                total_roi_seeds+=len(chosen);inferred_sources+=bool(chosen)
                if extra_models:
                    all_models=case['models']+extra_models
                    native=proposals(case['models'][0],all_models,current)
                    native=[native[i] for i in valid_boxes([p['box_xyxy'] for p in native],mask)]
                    native=attach(native,all_models,case['source_sha256'],image.shape[:2])
                    if native:
                        boxes=[p['box_xyxy'] for p in native];vectors=paired_features(embeddings(encoder,image,boxes),embeddings(encoder,expected,boxes))
                        with torch.inference_mode():final_scores=head(vectors).softmax(1).tolist()
            trial=select(current,native,final_scores,HEAD_SHA);final_proposals+=len(native)
            save(folder/(Path(name).stem+'_predictions.json'),dict(image=name,current=current,trial=trial,seeds=seeds,seed_probabilities=seed_scores,
                chosen_seeds=chosen,ROI_evidence=roi_evidence,extra_models=extra_models,proposals=native,probabilities=final_scores,
                alignment=case['alignment'],head_sha256=HEAD_SHA,source_sha256=case['source_sha256']))
            entry=entrymap[name];targets=read_targets('train',name,[2736,3648],entry['label_sha256'],pins)
            a=matches(current['all_predictions'],targets)[0];b=matches(trial['all_predictions'],targets)[0]
            rows.append(dict(image=name,current=metric(current['all_predictions'],targets),trial=metric(trial['all_predictions'],targets),gained=sorted(b-a),lost=sorted(a-b),ROI_seeds=len(chosen)))
        totals={v:{k:sum(r[v][k] for r in rows) for k in ('tp','unmatched','fn','predictions','targets')} for v in ('current','trial')}
        assert totals['current']['tp']==298 and totals['current']['unmatched']==4
        normal=sum(r['trial']['predictions'] for r in rows if r['image'].startswith('normal_'))
        qualifies=totals['trial']['tp']>298 and totals['trial']['unmatched']<=4 and not any(r['lost'] for r in rows) and normal==0
        summary=dict(qualifies=qualifies,summary=totals,normal_cues=normal,prescreen_seeds=total_seed_pairs,ROI_seeds=total_roi_seeds,ROI_sources=inferred_sources,final_proposals=final_proposals)
        save(folder/'report.json',dict(status='complete',**summary,cases=rows))
        assert all(sha(Path(p))==v for p,v in pins.items()) and native_pose_runtime_fingerprint(REPO)==frozen and CONFIG==original_config
        result=dict(status='source_pass_requires_combined_fresh_holdouts_and_actual_gates' if qualifies else 'rejected',summary=summary,
            pins=pins,runtime=frozen,no_training=True,no_validation_read=True,no_deployment=True,field_accuracy=False,seconds=round(time.monotonic()-started,2))
        save(OUT/'report.json',result);save(OUT/'progress.json',dict(status=result['status'],seconds=result['seconds']));print(dict(status=result['status'],summary=summary),flush=True)
    except BaseException as error:
        save(OUT/'progress.json',dict(status='failed',error=type(error).__name__+': '+str(error),seconds=round(time.monotonic()-started,2)));raise


if __name__=='__main__':main()
