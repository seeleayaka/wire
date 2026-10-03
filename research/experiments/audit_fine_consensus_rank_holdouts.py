"""All held-out source pixels, fresh fine outputs, ranking and final GT audit."""
import copy
import json
import sys
import time
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,REPO,DATA,load,save,sha,read_image,REFERENCE_SHA
from current_port_baseline_audit import BASE,read_current_case,read_targets
from inspection_agent.paired_native_pose import native_pose_runtime_fingerprint,HEAD_SHA,HEAD_RELATIVE
from inspection_agent.teacher_student_port_support import TEACHER_SHA,STUDENT_SHA
from inspection_agent.paired_port_features import expected_in_source,valid_boxes,embeddings,paired_features
from raw_pose_consensus import proposals
from audit_fine_consensus_rank import iou,independent_selection
from audit_port_multiscale_acceptance import metric,matches
SOURCE=ROOT/'artifacts/fine_consensus_rank_holdouts_20261004'
OUT=ROOT/'artifacts/fine_consensus_rank_holdouts_replay_20261004'
CURRENT=ROOT/'artifacts/paired_pose_native_three_20261004'
PREP=ROOT/'artifacts/paired_support_graph_20261003/source_selections'


def save(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    for attempt in range(8):
        try:tmp.replace(path);return
        except PermissionError:
            if attempt==7:raise
            time.sleep(.05)


def main():
    import cv2
    import numpy as np
    import torch
    cv2.setNumThreads(1);torch.set_num_threads(2)
    if OUT.exists():raise FileExistsError('Preserve full held-out audit')
    report=load(SOURCE/'report.json');digest=sha(SOURCE/'report.json');pins=report['pins']
    assert report['status'] in ('rejected','holdout_pass_requires_actual_reference_ROI_and_Qt_SAM')
    assert all(sha(Path(p))==v for p,v in pins.items()) and native_pose_runtime_fingerprint(REPO)==report['runtime']
    assert sha(REPO/HEAD_RELATIVE)==HEAD_SHA
    head=torch.nn.Linear(6144,3);head.load_state_dict(torch.load(REPO/HEAD_RELATIVE,map_location='cpu',weights_only=True)['state_dict']);head.eval().requires_grad_(False)
    referencepath=DATA/'images/train01/normal_073.JPG';assert sha(referencepath)==REFERENCE_SHA
    reference=read_image(referencepath);encoder=None;OUT.mkdir();stages={};total_cases=0;total_proposals=0
    for stage in report['stages']:
        entries=load(BASE/stage/'report.json')['cases'];assert len(entries)==(48 if stage=='inner' else 30)
        indexed={r['image']:r for r in load(PREP/stage/'index.json')['records']};rows=[];inferred=0;abstained=0
        for completed,entry in enumerate(entries):
            name=entry['image'];save(OUT/'progress.json',dict(status='running',stage=stage,completed=completed,total=len(entries)))
            case=load(SOURCE/stage/(Path(name).stem+'_predictions.json'));accepted=load(CURRENT/stage/(Path(name).stem+'_predictions.json'))
            assert case['current']==accepted['trial'] and case['head_sha256']==accepted['head_sha256']==HEAD_SHA
            sourcepath=DATA/'images'/('val01' if stage=='outer' else 'train01')/name;source_digest=sha(sourcepath)
            assert source_digest==case['source_sha256']==pins[str(sourcepath)]
            pair=load(Path(indexed[name]['path']));teacher,old=read_current_case(stage,entry,{})
            assert teacher==pair['teacher'];models=[teacher,pair['student'],pair['feature'],old['alternative']]
            remaining=5-(len(case['current']['all_predictions'])-len(case['current']['primary']))
            eligible=remaining>0 and any(p['confidence']>.05 for model in models for p in model['predictions']['merged_predictions'])
            assert eligible==case['eligible']
            views=case['new_views'];alignment=case['alignment'];recomputed=[];scores=[]
            if eligible and alignment['alignment_quality']['reliable']:
                assert len(views)==2;inferred+=1
                for view in views:
                    assert view['image']==name and view['source_sha256']==source_digest and view['view']=='fresh_fine960_stride720'
                    assert view['predictions']['source_shape']==[2736,3648]
                    assert len(view['predictions']['windows'])>0
                    assert all(window[2]-window[0]<=960 and window[3]-window[1]<=960 for window in view['predictions']['windows'])
                assert {v['weight_sha256'] for v in views}=={TEACHER_SHA,STUDENT_SHA}
                image=read_image(sourcepath);expected,mask=expected_in_source(reference,alignment['source_to_reference_homography'],image.shape[:2])
                generated=proposals(teacher,models+views,case['current'])
                generated=[generated[i] for i in valid_boxes([r['box_xyxy'] for r in generated],mask)]
                pool=[]
                for model in models+views:
                    assert model['source_sha256']==source_digest and model['predictions']['source_shape']==[2736,3648]
                    for p in model['predictions']['merged_predictions']:
                        l,t,r,b=p['box_xyxy']
                        if p['confidence']>.05 and 16<=l<r<=3632 and 16<=t<b<=2720:pool.append((model['weight_sha256'],p))
                for row in generated:
                    copied=copy.deepcopy(row);best={}
                    for weight,p in pool:
                        if p['class_id']==row['class_id']:
                            score=iou(row['box_xyxy'],p['box_xyxy'])
                            if score>=.5:best[weight]=max(score,best.get(weight,0.))
                    assert sorted(best)==row['semantic_model_vote_sha256'];copied['localization_voter_best_IoU']=best;recomputed.append(copied)
                assert len(recomputed)==len(case['proposals'])
                for a,b in zip(recomputed,case['proposals']):
                    qa=a.pop('localization_voter_best_IoU');qb=b['localization_voter_best_IoU'];clean=dict(b);clean.pop('localization_voter_best_IoU');assert a==clean
                    for weight in qa:assert abs(qa[weight]-qb[weight])<1e-12
                if generated:
                    import dino_feature_diff as dino
                    if encoder is None:encoder=dino._model();encoder.eval().requires_grad_(False);torch.set_num_threads(2)
                    boxes=[p['box_xyxy'] for p in generated]
                    vectors=paired_features(embeddings(encoder,image,boxes),embeddings(encoder,expected,boxes))
                    with torch.inference_mode():scores=head(vectors).softmax(1).tolist()
                    np.testing.assert_allclose(scores,case['probabilities'],atol=1e-6,rtol=1e-5)
            else:
                assert not views and not case['proposals'] and not case['probabilities']
                if eligible:abstained+=1;assert case['skip_reason']=='unreliable_original_SIFT'
            trial=independent_selection(case['current'],case['proposals'],case['probabilities'],HEAD_SHA);assert trial==case['trial']
            targets=read_targets(stage,name,[2736,3648],entry['label_sha256'],{})
            oldhits=matches(case['current']['all_predictions'],targets)[0];newhits=matches(trial['all_predictions'],targets)[0]
            rows.append(dict(image=name,current=metric(case['current']['all_predictions'],targets),trial=metric(trial['all_predictions'],targets),gained=sorted(newhits-oldhits),lost=sorted(oldhits-newhits),skip_reason=case['skip_reason']))
            total_cases+=1;total_proposals+=len(case['proposals'])
        totals={v:{k:sum(r[v][k] for r in rows) for k in ('tp','unmatched','fn','predictions','targets')} for v in ('current','trial')}
        base_tp,base_fp=(68,0) if stage=='inner' else (40,1);assert totals['current']['tp']==base_tp and totals['current']['unmatched']==base_fp
        normal=sum(r['trial']['predictions'] for r in rows if r['image'].startswith('normal_'))
        qualifies=(totals['trial']['tp']>base_tp if stage=='inner' else totals['trial']['tp']>=base_tp) and totals['trial']['unmatched']<=base_fp and not any(r['lost'] for r in rows) and normal==0
        summary=dict(qualifies=qualifies,summary=totals,normal_cues=normal,fresh_fine_sources=inferred,registration_abstentions=abstained)
        assert summary==report['stages'][stage];stages[stage]=summary
    assert all(sha(Path(p))==v for p,v in pins.items()) and sha(SOURCE/'report.json')==digest and native_pose_runtime_fingerprint(REPO)==report['runtime']
    result=dict(status='pass',experiment_status=report['status'],sources=total_cases,proposals=total_proposals,stages=stages,
        original_pixel_DINO_head_scores_recomputed=True,independent_voter_IoU_rank_novelty_GT_replay=True,
        detector_predictions_cached_not_reinferred=True,source_report_sha256=digest,auditor_sha256=sha(Path(__file__)),no_deployment=True,field_accuracy=False)
    save(OUT/'report.json',result);save(OUT/'progress.json',dict(status='pass'));print(result,flush=True)


if __name__=='__main__':main()
