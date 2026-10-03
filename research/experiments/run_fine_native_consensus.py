"""Reuse real fine-tile detections, three-checkpoint vote and accepted head."""
import os
import sys
import time
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,REPO,DATA,load,save,sha,read_image,REFERENCE_SHA
from current_port_baseline_audit import BASE,read_current_case,read_targets
from inspection_agent.paired_native_pose import native_pose_runtime_fingerprint,HEAD_SHA,HEAD_RELATIVE
from inspection_agent.paired_port_features import expected_in_source,valid_boxes,embeddings,paired_features
from audit_port_multiscale_acceptance import metric,matches
from raw_pose_consensus import proposals,select
SOURCE=ROOT/'artifacts/paired_fine_tile_views_20261003/full'
CURRENT=ROOT/'artifacts/paired_pose_native_three_20261004/full_train'
PREP=ROOT/'artifacts/paired_support_graph_20261003/source_selections/train/index.json'
OUT=ROOT/'artifacts/fine_native_consensus_20261004'
PLAN=ROOT/'artifacts/fine_native_consensus_preregistration_20261004/PLAN.md'

def main():
    if OUT.exists():raise FileExistsError('Preserve one fine consensus experiment')
    import torch
    import cv2
    import dino_feature_diff as dino
    torch.set_num_threads(2);cv2.setNumThreads(1)
    prior=load(SOURCE/'report.json');assert prior['status']=='rejected' and prior['stage']=='train'
    frozen=native_pose_runtime_fingerprint(REPO);assert prior['pins'][str(REPO/'output/paired_port_geometry_20261003/last_head.pt')]==frozen['accepted']['head']
    OUT.mkdir();started=time.monotonic();referencepath=DATA/'images/train01/normal_073.JPG';assert sha(referencepath)==REFERENCE_SHA
    pins={str(p):sha(p) for p in (Path(__file__),PLAN,Path(__file__).with_name('raw_pose_consensus.py'),SOURCE/'report.json',PREP,referencepath,REPO/HEAD_RELATIVE)}
    assert pins[str(REPO/HEAD_RELATIVE)]==HEAD_SHA
    indexed={r['image']:r for r in load(PREP)['records']};entries=load(BASE/'train/report.json')['cases'];prepared=[]
    assert len(entries)==192
    # Generate complete candidate population without labels first.
    for entry in entries:
        name=entry['image'];stem=Path(name).stem;path=Path(indexed[name]['path']);assert sha(path)==indexed[name]['sha256'];pins[str(path)]=sha(path);paired=load(path)
        teacher,old=read_current_case('train',entry,pins);assert teacher==paired['teacher']
        finepath=SOURCE/'train'/(stem+'_predictions.json');assert sha(finepath) or finepath.exists()
        pins[str(finepath)]=sha(finepath);fine=load(finepath)
        source=DATA/'images/train01'/name;assert sha(source)==prior['pins'][str(source)]
        for view in fine['new_views']:assert view['source_sha256']==sha(source) and view['weight_sha256'] in (frozen['accepted']['accepted']['accepted']['accepted_feature']['accepted']['teacher'],frozen['accepted']['accepted']['accepted']['accepted_feature']['accepted']['student'])
        currentpath=CURRENT/(stem+'_predictions.json');pins[str(currentpath)]=sha(currentpath);current=load(currentpath)['trial']
        remaining=5-(len(current['all_predictions'])-len(current['primary']));assert remaining>=0
        rows=proposals(teacher,[teacher,paired['student'],paired['feature'],old['alternative'],*fine['new_views']],current) if remaining else []
        prepared.append((entry,current,rows,fine['alignment']))
    save(OUT/'protocol.json',dict(pins=pins,runtime=frozen,head_sha256=HEAD_SHA,cached_fine_views_actual=True,
        same_checkpoint_views_one_vote=True,no_training=True,no_sweeps=True,field_accuracy=False))
    reference=read_image(referencepath);encoder=None;head=torch.nn.Linear(6144,3)
    head.load_state_dict(torch.load(REPO/HEAD_RELATIVE,map_location='cpu',weights_only=True)['state_dict']);head.eval().requires_grad_(False)
    folder=OUT/'train';folder.mkdir();rows=[];proposal_count=0
    try:
        for i,(entry,current,native,alignment) in enumerate(prepared):
            name=entry['image'];scores=[];source=DATA/'images/train01'/name;pins[str(source)]=sha(source)
            save(OUT/'progress.json',dict(status='running',pid=os.getpid(),seconds=round(time.monotonic()-started,2),image=name,completed=i,total=192,proposals=proposal_count))
            if native and alignment and alignment.get('alignment_quality',{}).get('reliable'):
                image=read_image(source);expected,mask=expected_in_source(reference,alignment['source_to_reference_homography'],image.shape[:2])
                native=[native[j] for j in valid_boxes([r['box_xyxy'] for r in native],mask)]
                if native:
                    if encoder is None:encoder=dino._model();encoder.eval().requires_grad_(False);torch.set_num_threads(2)
                    boxes=[r['box_xyxy'] for r in native];vectors=paired_features(embeddings(encoder,image,boxes),embeddings(encoder,expected,boxes))
                    with torch.inference_mode():scores=head(vectors).softmax(dim=1).tolist()
            else:native=[]
            trial=select(current,native,scores,HEAD_SHA);proposal_count+=len(native)
            save(folder/(Path(name).stem+'_predictions.json'),dict(image=name,current=current,trial=trial,proposals=native,probabilities=scores,alignment=alignment))
            targets=read_targets('train',name,[2736,3648],entry['label_sha256'],pins)
            oldhits=matches(current['all_predictions'],targets)[0];newhits=matches(trial['all_predictions'],targets)[0]
            rows.append(dict(image=name,current=metric(current['all_predictions'],targets),trial=metric(trial['all_predictions'],targets),gained=sorted(newhits-oldhits),lost=sorted(oldhits-newhits)))
        totals={version:{k:sum(row[version][k] for row in rows) for k in ('tp','unmatched','fn','predictions','targets')} for version in ('current','trial')}
        normal=sum(row['trial']['predictions'] for row in rows if row['image'].startswith('normal_'))
        qualifies=totals['trial']['tp']>295 and totals['trial']['unmatched']<=4 and normal==0 and not any(row['lost'] for row in rows)
        summary=dict(qualifies=qualifies,summary=totals,normal_cues=normal,valid_proposals=proposal_count)
        save(folder/'report.json',dict(status='complete',**summary,cases=rows))
        assert all(sha(Path(p))==v for p,v in pins.items()) and native_pose_runtime_fingerprint(REPO)==frozen
        result=dict(status='source_pass_requires_fresh_fine_holdouts_and_actual_gates' if qualifies else 'rejected',stages=dict(train=summary),pins=pins,runtime=frozen,
            no_deployment=True,no_training=True,field_accuracy=False,seconds=round(time.monotonic()-started,2))
        save(OUT/'report.json',result);save(OUT/'progress.json',dict(status=result['status'],seconds=result['seconds']));print(dict(status=result['status'],**summary),flush=True)
    except BaseException as error:
        save(OUT/'progress.json',dict(status='failed',error=type(error).__name__+': '+str(error)));raise

if __name__=='__main__':main()
