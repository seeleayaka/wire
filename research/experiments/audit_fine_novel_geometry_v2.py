"""Independently replay ALL192 novelty selection and fine relative coverage."""
import copy
import json
import time
import sys
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,REPO,DATA,load,save,sha,read_image,REFERENCE_SHA
from current_port_baseline_audit import BASE,read_current_case,read_targets
from inspection_agent.paired_native_pose import native_pose_runtime_fingerprint,HEAD_SHA
from inspection_agent.paired_native_pose_features import select as old_select
from inspection_agent.paired_port_features import expected_in_source,valid_boxes
from audit_port_multiscale_acceptance import metric,matches
from audit_fine_consensus_rank import duplicate,iou
SOURCE=ROOT/'artifacts/fine_novel_geometry_20261004'
OUT=ROOT/'artifacts/fine_novel_geometry_replay_20261004_v2'
FINE=ROOT/'artifacts/fine_native_consensus_20261004/train'
PREP=ROOT/'artifacts/paired_support_graph_20261003/source_selections/train/index.json'
VIEWS=ROOT/'artifacts/paired_fine_tile_views_20261003/full/train'
REGRESSION=ROOT/'artifacts/paired_box_regression_geometry_20261004'
TRAINING=ROOT/'artifacts/fine_pose_training_20261004/training'


def save(path,value):
    # Bounded retry for Windows scanner sharing locks, never ignore a failure.
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    for attempt in range(8):
        try:
            tmp.replace(path);return
        except PermissionError:
            if attempt==7:raise
            time.sleep(.05)


def independent_novelty(current,proposals,scores):
    indices=[i for i,row in enumerate(proposals) if not duplicate(row['box_xyxy'],current['all_predictions'])];excluded=set()
    for _ in range(len(indices)+1):
        kept_indices=[i for i in indices if (proposals[i]['class_id'],*proposals[i]['box_xyxy']) not in excluded]
        result=old_select(current,[proposals[i] for i in kept_indices],[scores[i] for i in kept_indices],HEAD_SHA)
        prior=[];bad=None
        for row in result['paired_semantic_additions']:
            if duplicate(row['box_xyxy'],prior):bad=row;break
            prior.append(row)
        if bad is None:return result
        excluded.add((bad['class_id'],*bad['box_xyxy']))
    raise AssertionError('Novelty refill did not converge')


def main():
    import cv2
    import numpy as np
    import torch
    torch.set_num_threads(2);cv2.setNumThreads(1)
    if OUT.exists():raise FileExistsError('Preserve novelty geometry audit')
    report=load(SOURCE/'report.json');digest=sha(SOURCE/'report.json');pins=report['pins']
    assert report['status']=='rejected_no_new_geometry_potential' and not (SOURCE/'fresh_semantics').exists()
    assert all(sha(Path(p))==v for p,v in pins.items()) and native_pose_runtime_fingerprint(REPO)==report['runtime']
    raw=torch.load(TRAINING/'raw_features.pt',map_location='cpu',weights_only=True)['features'];records=load(TRAINING/'raw_samples.json')
    folds=torch.tensor([r['fold'] for r in records]);predictions=torch.zeros((969,4));heads={}
    for fold in range(3):
        path=REGRESSION/'heads_oof'/('fold'+str(fold)+'.pt');assert sha(path)==pins[str(path)];heads[fold]=sha(path)
        state=torch.load(path,map_location='cpu',weights_only=True)
        with torch.inference_mode():predictions[folds==fold]=torch.nn.functional.linear(raw[folds==fold],state['weight'],state['bias'])
    sourcefolds=load(SOURCE/'protocol.json')['source_folds'];indexed={r['image']:r for r in load(PREP)['records']}
    reference=read_image(DATA/'images/train01/normal_073.JPG');assert sha(DATA/'images/train01/normal_073.JPG')==REFERENCE_SHA
    OUT.mkdir();rows=[];coverage=[];offset=0
    for completed,entry in enumerate(load(BASE/'train/report.json')['cases']):
        name=entry['image'];save(OUT/'progress.json',dict(status='running',completed=completed,total=192))
        case=load(FINE/(Path(name).stem+'_predictions.json'));saved=load(SOURCE/'duplicate_only'/(Path(name).stem+'_predictions.json'))
        assert case['current']==saved['current'] and case['proposals']==saved['proposals'] and case['probabilities']==saved['probabilities']
        trial=independent_novelty(saved['current'],saved['proposals'],saved['probabilities']);assert trial==saved['trial']
        targets=read_targets('train',name,[2736,3648],entry['label_sha256'],{})
        old=matches(case['current']['all_predictions'],targets)[0];new=matches(trial['all_predictions'],targets)[0]
        rows.append(dict(image=name,current=metric(case['current']['all_predictions'],targets),trial=metric(trial['all_predictions'],targets),gained=sorted(new-old),lost=sorted(old-new)))
        geometry=load(SOURCE/'geometry'/(Path(name).stem+'_geometry.json'));assert geometry['current']==case['current'] and geometry['alignment']==case['alignment']
        candidates=case['proposals'];n=len(candidates);local=records[offset:offset+n];assert all(r['image']==name and r['box']==p['box_xyxy'] and r['fold']==sourcefolds[name] for r,p in zip(local,candidates))
        deltas=predictions[offset:offset+n].tolist();offset+=n;before=[];after=[]
        if candidates:
            pair=load(Path(indexed[name]['path']));teacher,oldcase=read_current_case('train',entry,{})
            models=[teacher,pair['student'],pair['feature'],oldcase['alternative'],*load(VIEWS/(Path(name).stem+'_predictions.json'))['new_views']]
            pool=[]
            for model in models:
                for p in model['predictions']['merged_predictions']:
                    l,t,r,b=p['box_xyxy']
                    if p['confidence']>.05 and 16<=l<r<=3632 and 16<=t<b<=2720:pool.append((model['weight_sha256'],p))
            _,mask=expected_in_source(reference,case['alignment']['source_to_reference_homography'],(2736,3648))
            def eligible(row):
                l,t,r,b=row['box_xyxy']
                if not (16<=l<r<=3632 and 16<=t<b<=2720) or duplicate(row['box_xyxy'],case['current']['all_predictions']):return False
                votes={digest for digest,p in pool if p['class_id']==row['class_id'] and iou(p['box_xyxy'],row['box_xyxy'])>=.5}
                row['semantic_model_vote_sha256']=sorted(votes)
                return len(votes)>=3 and valid_boxes([row['box_xyxy']],mask)==[0]
            for row,delta in zip(candidates,deltas):
                original=copy.deepcopy(row)
                if eligible(original):before.append(original)
                # Independent centre/size decoding, not the training decode helper.
                l,t,r,b=row['box_xyxy'];w,h=r-l,b-t;dx,dy=max(-.15,min(.15,delta[0])),max(-.15,min(.15,delta[1]))
                sw,sh=np.exp(np.clip(delta[2:],np.log(.8),np.log(1.2)));cx,cy=(l+r)/2+dx*w,(t+b)/2+dy*h
                refined=copy.deepcopy(row);refined.update(box_xyxy=[cx-w*sw/2,cy-h*sh/2,cx+w*sw/2,cy+h*sh/2],
                    relative_box_prediction=delta,regression_head_sha256=heads[sourcefolds[name]],relative_box_refined_from=row['box_xyxy'])
                if eligible(refined):after.append(refined)
            assert len(before)==len(geometry['original_eligible']) and len(after)==len(geometry['refined_eligible'])
            for a,b in zip(before,geometry['original_eligible']):assert a==b
            for a,b in zip(after,geometry['refined_eligible']):
                np.testing.assert_allclose(a['box_xyxy'],b['box_xyxy'],atol=1e-9,rtol=1e-12)
                a['box_xyxy']=b['box_xyxy'];assert a==b
        missing=set(range(len(targets)))-old
        cover=lambda pool:sorted(i for i in missing if any(p['class_id']==targets[i]['class_id'] and iou(p['box_xyxy'],targets[i]['box'])>=.5 for p in pool))
        a=cover(before);b=cover(before+after);summary=dict(image=name,original_coverage_targets=a,union_coverage_targets=b,
            new_potential_targets=sorted(set(b)-set(a)),missed_targets=len(missing),original_eligible=len(before),refined_eligible=len(after))
        assert summary==geometry['summary'];coverage.append(summary)
    totals={v:{k:sum(r[v][k] for r in rows) for k in ('tp','unmatched','fn','predictions','targets')} for v in ('current','trial')}
    assert totals==report['stages']['duplicate_only']['summary'] and totals['trial']['tp']==297 and totals['trial']['unmatched']==5 and offset==969
    summary=dict(missed_targets=sum(r['missed_targets'] for r in coverage),original_possible_coverage=sum(len(r['original_coverage_targets']) for r in coverage),
        union_possible_coverage=sum(len(r['union_coverage_targets']) for r in coverage),newly_possible_targets=sum(len(r['new_potential_targets']) for r in coverage))
    assert summary==report['stages']['geometry_potential']
    assert all(sha(Path(p))==v for p,v in pins.items()) and sha(SOURCE/'report.json')==digest and native_pose_runtime_fingerprint(REPO)==report['runtime']
    result=dict(status='pass',sources=192,fine_proposals=969,source_report_sha256=digest,duplicate_summary=totals,geometry_summary=summary,
        independent_IoMin_selector_relative_decode_vote_and_GT_replay=True,fresh_semantics_not_run=True,no_deployment=True,field_accuracy=False)
    save(OUT/'report.json',result);save(OUT/'progress.json',dict(status='pass'));print(result,flush=True)


if __name__=='__main__':main()


