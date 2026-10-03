"""Finite ALL192 duplicate/relative fine-box test with fresh final-geometry scores."""
import copy
import os
import sys
import time
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,REPO,DATA,load,save,sha,read_image,REFERENCE_SHA
from current_port_baseline_audit import BASE,read_current_case,read_targets
from audit_port_multiscale_acceptance import metric,matches,overlap
from inspection_agent.paired_native_pose import native_pose_runtime_fingerprint,HEAD_SHA,HEAD_RELATIVE
from inspection_agent.paired_port_features import expected_in_source,valid_boxes,embeddings,paired_features
from relative_port_box import decode
from novel_box_geometry import select,is_duplicate
SOURCE=ROOT/'artifacts/fine_native_consensus_20261004/train'
REGRESSION=ROOT/'artifacts/paired_box_regression_geometry_20261004'
VIEWS=ROOT/'artifacts/paired_fine_tile_views_20261003/full/train'
PREP=ROOT/'artifacts/paired_support_graph_20261003/source_selections/train/index.json'
TRAINING=ROOT/'artifacts/fine_pose_training_20261004/training'
OUT=ROOT/'artifacts/fine_novel_geometry_20261004'
PLAN=ROOT/'artifacts/fine_novel_geometry_preregistration_20261004/PLAN.md'


def main():
    import cv2
    import numpy as np
    import torch
    import psutil
    cv2.setNumThreads(1);torch.set_num_threads(2)
    if OUT.exists():raise FileExistsError('Preserve finite novelty geometry test')
    assert psutil.virtual_memory().available>6*2**30,'Do not compete for inference memory'
    frozen=native_pose_runtime_fingerprint(REPO)
    oldreport=ROOT/'artifacts/fine_native_consensus_20261004/report.json'
    report=load(oldreport);assert report['status']=='rejected' and report['runtime']==frozen
    regression=load(REGRESSION/'report.json');assert regression['runtime']==frozen and regression['status']=='rejected_no_new_geometry_potential'
    ra=ROOT/'artifacts/paired_box_regression_geometry_replay_20261004/report.json'
    assert load(ra)['status']=='pass' and load(ra)['source_report_sha256']==sha(REGRESSION/'report.json')
    assert all(sha(Path(p))==v for p,v in regression['pins'].items())
    groupspath=ROOT/'artifacts/port_training_multiscale_20261002/protocol.json'
    names=sorted(load(groupspath)['train_sources']);folds={n:i%3 for i,n in enumerate(names)};assert len(names)==192
    referencepath=DATA/'images/train01/normal_073.JPG';assert sha(referencepath)==REFERENCE_SHA
    pins={str(p):sha(p) for p in (Path(__file__),PLAN,Path(__file__).with_name('novel_box_geometry.py'),
        Path(__file__).with_name('relative_port_box.py'),oldreport,REGRESSION/'report.json',ra,groupspath,referencepath,
        PREP,TRAINING/'raw_features.pt',TRAINING/'raw_samples.json',REPO/HEAD_RELATIVE,
        REPO/'inspection_agent/paired_port_features.py',REPO/'inspection_agent/paired_native_pose_features.py')}
    assert pins[str(REPO/HEAD_RELATIVE)]==HEAD_SHA
    raw=torch.load(TRAINING/'raw_features.pt',map_location='cpu',weights_only=True)['features']
    raw_records=load(TRAINING/'raw_samples.json');assert raw.shape==(969,6144)
    rawfolds=torch.tensor([r['fold'] for r in raw_records]);predictions=torch.zeros((969,4));head_sha={}
    for fold in range(3):
        path=REGRESSION/'heads_oof'/('fold'+str(fold)+'.pt');pins[str(path)]=sha(path);head_sha[fold]=sha(path)
        head=torch.nn.Linear(6144,4);head.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));head.eval().requires_grad_(False)
        with torch.inference_mode():predictions[rawfolds==fold]=head(raw[rawfolds==fold])
    OUT.mkdir();started=time.monotonic();stages={}
    progress=lambda **kw:save(OUT/'progress.json',dict(status='running',pid=os.getpid(),seconds=round(time.monotonic()-started,2),**kw))
    save(OUT/'protocol.json',dict(pins=pins,runtime=frozen,train_sources=names,source_folds=folds,
        duplicate_intersection_over_smaller_threshold=.5,no_new_training=True,no_threshold_sweep=True,
        regression_only_source_OOF=True,no_deployment=True,field_accuracy=False))
    indexed={r['image']:r for r in load(PREP)['records']};entries=load(BASE/'train/report.json')['cases']
    assert sorted(r['image'] for r in entries)==names
    def summarize(rows):
        totals={v:{k:sum(r[v][k] for r in rows) for k in ('tp','unmatched','fn','predictions','targets')} for v in ('current','trial')}
        assert totals['current']['tp']==295 and totals['current']['unmatched']==4
        normal=sum(r['trial']['predictions'] for r in rows if r['image'].startswith('normal_'))
        qualifies=totals['trial']['tp']>295 and totals['trial']['unmatched']<=4 and normal==0 and not any(r['lost'] for r in rows)
        return dict(qualifies=qualifies,summary=totals,normal_cues=normal)
    def finish(status):
        assert all(sha(Path(p))==v for p,v in pins.items()) and native_pose_runtime_fingerprint(REPO)==frozen
        result=dict(status=status,stages=stages,pins=pins,runtime=frozen,no_new_training=True,
            no_validation_read=True,no_deployment=True,field_accuracy=False,seconds=round(time.monotonic()-started,2))
        save(OUT/'report.json',result);save(OUT/'progress.json',dict(status=status,seconds=result['seconds']))
        print(dict(status=status,stages=stages,seconds=result['seconds']),flush=True)
    def scores_row(entry,current,trial,targets):
        old=matches(current['all_predictions'],targets)[0];new=matches(trial['all_predictions'],targets)[0]
        return dict(image=entry['image'],current=metric(current['all_predictions'],targets),trial=metric(trial['all_predictions'],targets),gained=sorted(new-old),lost=sorted(old-new))
    try:
        duplicate=OUT/'duplicate_only';duplicate.mkdir();rows=[]
        for entry in entries:
            name=entry['image'];path=SOURCE/(Path(name).stem+'_predictions.json');pins[str(path)]=sha(path);case=load(path)
            trial=select(case['current'],case['proposals'],case['probabilities'],HEAD_SHA)
            targets=read_targets('train',name,[2736,3648],entry['label_sha256'],pins)
            rows.append(scores_row(entry,case['current'],trial,targets))
            save(duplicate/path.name,dict(image=name,current=case['current'],trial=trial,proposals=case['proposals'],probabilities=case['probabilities'],head_sha256=HEAD_SHA))
        summary=summarize(rows);stages['duplicate_only']=summary;save(duplicate/'report.json',dict(status='complete',**summary,cases=rows));print(dict(stage='duplicate_only',**summary),flush=True)
        # Geometry phase is predeclared, not a response to a selected photo's GT.
        reference=read_image(referencepath);geometry=OUT/'geometry';geometry.mkdir();coverage=[];offset=0
        for completed,entry in enumerate(entries):
            name=entry['image'];progress(phase='all192_fine_geometry_coverage',image=name,completed=completed,total=192)
            path=SOURCE/(Path(name).stem+'_predictions.json');case=load(path);candidates=case['proposals'];current=case['current'];n=len(candidates)
            local=raw_records[offset:offset+n];assert len(local)==n and all(r['image']==name and r['box']==p['box_xyxy'] and r['fold']==folds[name] for r,p in zip(local,candidates))
            deltas=predictions[offset:offset+n].tolist();offset+=n
            source=DATA/'images/train01'/name;pins[str(source)]=sha(source)
            before=[];after=[]
            if candidates:
                pairedpath=Path(indexed[name]['path']);assert sha(pairedpath)==indexed[name]['sha256'];pins[str(pairedpath)]=sha(pairedpath);paired=load(pairedpath)
                teacher,old=read_current_case('train',entry,pins);assert teacher==paired['teacher']
                finepath=VIEWS/(Path(name).stem+'_predictions.json');pins[str(finepath)]=sha(finepath);fine=load(finepath)
                models=[teacher,paired['student'],paired['feature'],old['alternative'],*fine['new_views']];pool=[]
                for model in models:
                    assert model['source_sha256']==pins[str(source)] and model['predictions']['source_shape']==[2736,3648]
                    for p in model['predictions']['merged_predictions']:
                        l,t,r,b=p['box_xyxy']
                        if p['confidence']>.05 and 16<=l<r<=3632 and 16<=t<b<=2720:pool.append((model['weight_sha256'],p))
                assert case['alignment']['alignment_quality']['reliable']
                image=read_image(source);_,mask=expected_in_source(reference,case['alignment']['source_to_reference_homography'],image.shape[:2])
                def eligible(p):
                    l,t,r,b=p['box_xyxy']
                    if not (16<=l<r<=3632 and 16<=t<b<=2720) or is_duplicate(p['box_xyxy'],current['all_predictions']):return False
                    votes={digest for digest,row in pool if row['class_id']==p['class_id'] and overlap(row['box_xyxy'],p['box_xyxy'])>=.5}
                    p['semantic_model_vote_sha256']=sorted(votes)
                    return len(votes)>=3 and valid_boxes([p['box_xyxy']],mask)==[0]
                for p,delta in zip(candidates,deltas):
                    original=copy.deepcopy(p)
                    if eligible(original):before.append(original)
                    refined=copy.deepcopy(p);refined.update(box_xyxy=decode(p['box_xyxy'],delta),relative_box_prediction=delta,
                        regression_head_sha256=head_sha[folds[name]],relative_box_refined_from=p['box_xyxy'])
                    if eligible(refined):after.append(refined)
            targets=read_targets('train',name,[2736,3648],entry['label_sha256'],pins);missing=set(range(len(targets)))-matches(current['all_predictions'],targets)[0]
            cover=lambda pool:sorted(t for t in missing if any(p['class_id']==targets[t]['class_id'] and overlap(p['box_xyxy'],targets[t]['box'])>=.5 for p in pool))
            a=cover(before);b=cover(before+after)
            row=dict(image=name,original_coverage_targets=a,union_coverage_targets=b,new_potential_targets=sorted(set(b)-set(a)),missed_targets=len(missing),original_eligible=len(before),refined_eligible=len(after))
            coverage.append(row);save(geometry/(Path(name).stem+'_geometry.json'),dict(image=name,current=current,original_eligible=before,
                refined_eligible=after,summary=row,alignment=case['alignment'],regression_head_sha256=head_sha[folds[name]]))
        assert offset==969
        summary=dict(missed_targets=sum(r['missed_targets'] for r in coverage),original_possible_coverage=sum(len(r['original_coverage_targets']) for r in coverage),
            union_possible_coverage=sum(len(r['union_coverage_targets']) for r in coverage),newly_possible_targets=sum(len(r['new_potential_targets']) for r in coverage))
        assert summary['missed_targets']==49
        stages['geometry_potential']=summary;save(geometry/'report.json',dict(status='complete',summary=summary,cases=coverage,not_recognition_accuracy=True))
        print(dict(stage='geometry_potential',summary=summary),flush=True)
        if not summary['newly_possible_targets']:
            finish('duplicate_only_source_pass_requires_fresh_holds' if stages['duplicate_only']['qualifies'] else 'rejected_no_new_geometry_potential');return
        semantic=OUT/'fresh_semantics';semantic.mkdir();head=torch.nn.Linear(6144,3)
        head.load_state_dict(torch.load(REPO/HEAD_RELATIVE,map_location='cpu',weights_only=True)['state_dict']);head.eval().requires_grad_(False)
        encoder=None;rows=[]
        for completed,entry in enumerate(entries):
            name=entry['image'];progress(phase='original_pixel_final_geometry_semantics',image=name,completed=completed,total=192)
            case=load(geometry/(Path(name).stem+'_geometry.json'));current=case['current'];proposals=case['original_eligible']+case['refined_eligible']
            scores=[]
            if proposals:
                import dino_feature_diff as dino
                if encoder is None:encoder=dino._model();encoder.eval().requires_grad_(False);torch.set_num_threads(2)
                image=read_image(DATA/'images/train01'/name);expected,mask=expected_in_source(reference,case['alignment']['source_to_reference_homography'],image.shape[:2])
                boxes=[p['box_xyxy'] for p in proposals];assert valid_boxes(boxes,mask)==list(range(len(boxes)))
                vectors=paired_features(embeddings(encoder,image,boxes),embeddings(encoder,expected,boxes))
                with torch.inference_mode():scores=head(vectors).softmax(1).tolist()
            trial=select(current,proposals,scores,HEAD_SHA);targets=read_targets('train',name,[2736,3648],entry['label_sha256'],pins)
            rows.append(scores_row(entry,current,trial,targets))
            save(semantic/(Path(name).stem+'_predictions.json'),dict(image=name,current=current,trial=trial,proposals=proposals,probabilities=scores,head_sha256=HEAD_SHA))
        summary=summarize(rows);stages['fresh_semantics']=summary;save(semantic/'report.json',dict(status='complete',**summary,cases=rows));print(dict(stage='fresh_semantics',**summary),flush=True)
        finish('source_pass_requires_fresh_holdouts_and_actual_gates' if summary['qualifies'] else 'rejected_source_fresh_semantics')
    except BaseException as error:
        save(OUT/'progress.json',dict(status='failed',error=type(error).__name__+': '+str(error),seconds=round(time.monotonic()-started,2)));raise


if __name__=='__main__':main()
