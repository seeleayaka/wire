"""Independent voter-IoU/median/rank/novelty replay and frozen head/GT scoring."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,REPO,DATA,load,save,sha
from current_port_baseline_audit import BASE,read_current_case,read_targets
from inspection_agent.paired_native_pose import native_pose_runtime_fingerprint,HEAD_SHA,HEAD_RELATIVE
from inspection_agent.paired_port_features import select as append_original
from audit_port_multiscale_acceptance import metric,matches
SOURCE=ROOT/'artifacts/fine_consensus_rank_20261004'
FINE=ROOT/'artifacts/fine_native_consensus_20261004/train'
VIEWS=ROOT/'artifacts/paired_fine_tile_views_20261003/full/train'
PREP=ROOT/'artifacts/paired_support_graph_20261003/source_selections/train/index.json'
TRAINING=ROOT/'artifacts/fine_pose_training_20261004/training'
OUT=ROOT/'artifacts/fine_consensus_rank_replay_20261004'


def iou(a,b):
    inter=max(0.,min(a[2],b[2])-max(a[0],b[0]))*max(0.,min(a[3],b[3])-max(a[1],b[1]))
    return inter/max(1e-12,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)


def duplicate(box,others):
    for r in others:
        other=r['box_xyxy'];inter=max(0.,min(box[2],other[2])-max(box[0],other[0]))*max(0.,min(box[3],other[3])-max(box[1],other[1]))
        area=min((box[2]-box[0])*(box[3]-box[1]),(other[2]-other[0])*(other[3]-other[1]))
        if inter/area>=.5:return True
    return False


def independent_selection(current,proposals,probabilities,head):
    import numpy as np
    eligible=[]
    for row,prob in zip(proposals,probabilities):
        cls=int(np.argmax(prob))
        if cls==row['class_id']+1 and prob[cls]>=.98 and len(set(row['semantic_model_vote_sha256']))>=3 and not duplicate(row['box_xyxy'],current['all_predictions']):
            quality=float(np.median(list(row['localization_voter_best_IoU'].values())))
            eligible.append(((-quality,-prob[cls],*row['box_xyxy'],row['class_id']),row,prob))
    excluded=set()
    for _ in range(len(eligible)+1):
        chosen={}
        for rank,row,prob in eligible:
            key=(row['class_id'],*row['box_xyxy'])
            if key in excluded:continue
            parent=row['pose_parent_seed_id']
            if parent not in chosen or rank<chosen[parent][0]:chosen[parent]=(rank,row,prob)
        out=append_original(current,[r[1] for r in chosen.values()],[r[2] for r in chosen.values()],head)
        prior=[];bad=None
        for row in out['paired_semantic_additions']:
            if duplicate(row['box_xyxy'],prior):bad=row;break
            prior.append(row)
        if bad is None:return out
        excluded.add((bad['class_id'],*bad['box_xyxy']))
    raise AssertionError('Independent finite duplicate refill failed')


def main():
    import numpy as np
    import torch
    torch.set_num_threads(2)
    if OUT.exists():raise FileExistsError('Preserve full consensus ranking audit')
    report=load(SOURCE/'report.json');digest=sha(SOURCE/'report.json');pins=report['pins']
    assert report['status']=='source_pass_requires_fresh_fine_holdouts_and_actual_gates'
    assert all(sha(Path(p))==v for p,v in pins.items()) and native_pose_runtime_fingerprint(REPO)==report['runtime']
    upstream=ROOT/'artifacts/fine_pose_training_audit_20261004/report.json'
    assert load(upstream)['status']=='complete' and load(upstream)['source_report_sha256']==sha(ROOT/'artifacts/fine_pose_training_20261004/report.json')
    raw=torch.load(TRAINING/'raw_features.pt',map_location='cpu',weights_only=True)['features'];records=load(TRAINING/'raw_samples.json')
    assert raw.shape==(969,6144) and sha(REPO/HEAD_RELATIVE)==HEAD_SHA
    head=torch.nn.Linear(6144,3);head.load_state_dict(torch.load(REPO/HEAD_RELATIVE,map_location='cpu',weights_only=True)['state_dict']);head.eval().requires_grad_(False)
    with torch.inference_mode():scores=head(raw).softmax(1).numpy()
    indexed={r['image']:r for r in load(PREP)['records']};entries=load(BASE/'train/report.json')['cases'];OUT.mkdir();rows=[];offset=0
    for entry in entries:
        name=entry['image'];saved=load(SOURCE/'train'/(Path(name).stem+'_predictions.json'));fine=load(FINE/(Path(name).stem+'_predictions.json'))
        proposals=saved['proposals'];n=len(proposals);assert saved['current']==fine['current'] and saved['probabilities']==fine['probabilities']
        local=records[offset:offset+n];assert len(local)==n and all(r['image']==name and r['box']==p['box_xyxy'] for r,p in zip(local,proposals))
        np.testing.assert_allclose(scores[offset:offset+n],np.array(saved['probabilities']).reshape(n,3),atol=1e-6,rtol=1e-5);offset+=n
        if proposals:
            pair=load(Path(indexed[name]['path']));teacher,old=read_current_case('train',entry,{})
            models=[teacher,pair['student'],pair['feature'],old['alternative'],*load(VIEWS/(Path(name).stem+'_predictions.json'))['new_views']]
            for row,original in zip(proposals,fine['proposals']):
                clean=dict(row);values=clean.pop('localization_voter_best_IoU');assert clean==original
                best={}
                for model in models:
                    assert model['source_sha256']==sha(DATA/'images/train01'/name) and model['predictions']['source_shape']==[2736,3648]
                    for p in model['predictions']['merged_predictions']:
                        l,t,r,b=p['box_xyxy']
                        if p['class_id']==row['class_id'] and p['confidence']>.05 and 16<=l<r<=3632 and 16<=t<b<=2720:
                            value=iou(p['box_xyxy'],row['box_xyxy'])
                            if value>=.5:best[model['weight_sha256']]=max(value,best.get(model['weight_sha256'],0.))
                assert sorted(best)==row['semantic_model_vote_sha256'] and set(best)==set(values)
                for weight in best:assert abs(best[weight]-values[weight])<1e-12
        trial=independent_selection(saved['current'],proposals,saved['probabilities'],HEAD_SHA);assert trial==saved['trial']
        assert trial['all_predictions'][:len(saved['current']['all_predictions'])]==saved['current']['all_predictions']
        assert len(trial['primary'])<=5 and len(trial['all_predictions'])<=len(trial['primary'])+5
        added=trial['paired_semantic_additions'];assert all(not duplicate(r['box_xyxy'],saved['current']['all_predictions']+added[:j]) for j,r in enumerate(added))
        targets=read_targets('train',name,[2736,3648],entry['label_sha256'],{})
        oldhits=matches(saved['current']['all_predictions'],targets)[0];newhits=matches(trial['all_predictions'],targets)[0]
        rows.append(dict(image=name,current=metric(saved['current']['all_predictions'],targets),trial=metric(trial['all_predictions'],targets),gained=sorted(newhits-oldhits),lost=sorted(oldhits-newhits)))
    totals={v:{k:sum(r[v][k] for r in rows) for k in ('tp','unmatched','fn','predictions','targets')} for v in ('current','trial')}
    summary=dict(qualifies=totals['trial']['tp']>295 and totals['trial']['unmatched']<=4 and not any(r['lost'] for r in rows),summary=totals,normal_cues=sum(r['trial']['predictions'] for r in rows if r['image'].startswith('normal_')))
    assert offset==969 and summary==report['stages']['train'] and summary['normal_cues']==0
    assert all(sha(Path(p))==v for p,v in pins.items()) and sha(SOURCE/'report.json')==digest
    assert native_pose_runtime_fingerprint(REPO)==report['runtime']
    result=dict(status='pass',sources=192,candidates=969,summary=summary,independent_voter_IoU_median_rank_duplicate_and_GT_replay=True,
        cached_original_pixel_head_scores_recomputed=True,upstream_feature_audit_sha256=sha(upstream),source_report_sha256=digest,
        auditor_sha256=sha(Path(__file__)),no_new_detector_inference=True,no_deployment=True,field_accuracy=False)
    save(OUT/'report.json',result);print(result,flush=True)


if __name__=='__main__':main()
