"""Frozen new head only for new pose cues; preserve accepted native prefix."""
import os
import sys
import time
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,REPO,DATA,load,save,sha
from current_port_baseline_audit import BASE,read_targets
from audit_port_multiscale_acceptance import metric,matches
from paired_pose_search import select
from inspection_agent.paired_median_geometry import median_runtime_fingerprint
SOURCE=ROOT/'artifacts/paired_pose_native_training_20261004'
AUDIT=ROOT/'artifacts/paired_pose_native_training_audit_20261004/report.json'
OUT=ROOT/'artifacts/paired_pose_native_only_20261004'
PLAN=ROOT/'artifacts/paired_pose_native_only_preregistration_20261004/PLAN.md'


def summarize(folder,rows,pins,strict_gain):
    totals={v:{k:sum(r[v][k] for r in rows) for k in ('tp','unmatched','fn','predictions','targets')} for v in ('current','trial')}
    gain=totals['trial']['tp']>totals['current']['tp'] if strict_gain else totals['trial']['tp']>=totals['current']['tp']
    normal=sum(r['trial']['predictions'] for r in rows if r['image'].startswith('normal_'))
    qualifies=gain and totals['trial']['unmatched']<=totals['current']['unmatched'] and not any(r['lost'] for r in rows) and normal==0
    assert {p:sha(Path(p)) for p in pins}==pins
    result=dict(status='complete',qualifies=qualifies,summary=totals,cases=rows,normal_cues=normal)
    save(folder/'report.json',result);print(str(dict(stage=folder.name,qualifies=qualifies,summary=totals)),flush=True)
    return result


def cached_stage(stage,pins):
    dataset='train' if stage in ('source_train_oof','full_train') else stage
    folder=OUT/stage;folder.mkdir();rows=[]
    for entry in load(BASE/dataset/'report.json')['cases']:
        name=entry['image'];path=SOURCE/stage/(Path(name).stem+'_predictions.json');pins[str(path)]=sha(path);case=load(path)
        current=case['current'];trial=select(current,case['proposals'],case['auxiliary_probabilities'],case['auxiliary_head_sha256'])
        assert trial['all_predictions'][:len(current['all_predictions'])]==current['all_predictions']
        save(folder/path.name,dict(image=name,current=current,trial=trial,proposals=case['proposals'],
            probabilities=case['auxiliary_probabilities'],head_sha256=case['auxiliary_head_sha256'],cached_feature_provenance=str(path)))
        targets=read_targets(dataset,name,[2736,3648],entry['label_sha256'],pins)
        old,new=matches(current['all_predictions'],targets)[0],matches(trial['all_predictions'],targets)[0]
        rows.append(dict(image=name,current=metric(current['all_predictions'],targets),trial=metric(trial['all_predictions'],targets),gained=sorted(new-old),lost=sorted(old-new)))
    return summarize(folder,rows,pins,True)


def main():
    if OUT.exists():raise FileExistsError('Preserve new head only trial')
    source=load(SOURCE/'report.json');assert source['status']=='rejected' and source['stage']=='inner'
    assert load(AUDIT)['status']=='complete'
    pins=dict(source['pins']);pins.update(load(AUDIT)['pins'])
    for path in (Path(__file__),Path(__file__).with_name('paired_pose_search.py'),PLAN,AUDIT,SOURCE/'report.json'):pins[str(path)]=sha(path)
    assert {p:sha(Path(p)) for p in pins}==pins
    frozen=median_runtime_fingerprint(REPO);assert source['median_runtime_fingerprint']==frozen
    OUT.mkdir();started=time.monotonic();stages={}
    save(OUT/'protocol.json',dict(pins=pins,median_runtime_fingerprint=frozen,new_head_only_for_new_proposals=True,
        no_training=True,unchanged_p98=True,original_prefix_and_model_unchanged=True,reused_validation=True,no_deployment=True,field_accuracy=False))
    try:
        for stage in ('source_train_oof','full_train','inner'):
            save(OUT/'progress.json',dict(status='running',pid=os.getpid(),stage=stage))
            result=cached_stage(stage,pins);stages[stage]=dict(qualifies=result['qualifies'],summary=result['summary'])
            if not result['qualifies']:break
        else:
            result=fresh_outer(pins,started);stages['outer']=dict(qualifies=result['qualifies'],summary=result['summary'])
        failed=next((stage for stage,value in stages.items() if not value['qualifies']),None)
        assert {p:sha(Path(p)) for p in pins}==pins and median_runtime_fingerprint(REPO)==frozen
        result=dict(status='rejected' if failed else 'source_pass_requires_reference_ROI_SAM_Qt',stage=failed,stages=stages,pins=pins,
            median_runtime_fingerprint=frozen,seconds=round(time.monotonic()-started,2),no_deployment=True,field_accuracy=False)
        save(OUT/'report.json',result);save(OUT/'progress.json',dict(status=result['status'],stage=failed))
    except BaseException as error:
        save(OUT/'progress.json',dict(status='failed',error=type(error).__name__+': '+str(error)));raise


def fresh_outer(pins,started):
    import torch
    import cv2
    import run_paired_pose_native_training as original
    # Reuse the exact fresh-feature routine; only scope output, selector and
    # source stage list. Its paired classifier agreement is not used here.
    headpath=SOURCE/'full/last_head.pt';digest=sha(headpath);pins[str(headpath)]=digest
    head=torch.nn.Linear(6144,3);head.load_state_dict(torch.load(headpath,map_location='cpu',weights_only=True)['state_dict']);head.eval()
    from prepare_paired_port_semantics import read_image,cached_alignments,REFERENCE_SHA
    from paired_port_semantics import expected_in_source,valid_boxes,paired_features,CachedReferenceSIFT
    from port_semantic_verifier import embeddings
    import assembly_auto_review_robust_v3 as registration
    inputs=ROOT/'artifacts/paired_pose_search_inputs_20261003';median=ROOT/'artifacts/paired_median_current_head_20261003'
    referencepath=DATA/'images/train01/normal_073.JPG';assert sha(referencepath)==REFERENCE_SHA;pins[str(referencepath)]=REFERENCE_SHA
    reference=read_image(referencepath);alignments=cached_alignments();encoder=None;torch.set_num_threads(1);cv2.setNumThreads(1)
    folder=OUT/'outer';folder.mkdir();rows=[];entries=load(BASE/'outer/report.json')['cases'];assert len(entries)==30
    with CachedReferenceSIFT(reference):
        for index,entry in enumerate(entries):
            name=entry['image'];stem=Path(name).stem;path=inputs/('outer_'+stem+'_proposals.json');pins[str(path)]=sha(path);case=load(path)
            current=case['current'];native=case['extended_candidates'];scores=[];alignment=None
            save(OUT/'progress.json',dict(status='running',pid=os.getpid(),stage='outer',image=name,completed=index,total=30,seconds=round(time.monotonic()-started,2)))
            if native:
                source=DATA/'images/val01'/name;pins[str(source)]=sha(source);image=read_image(source)
                if name in alignments:
                    provenance,cached=alignments[name];pins[str(provenance)]=sha(provenance)
                    assert cached['image_fingerprints']['source_sha256']==pins[str(source)];alignment=cached['alignment']
                else:cv2.setRNGSeed(0);_,alignment=registration.automatic_homography(reference,image)
                if alignment.get('alignment_quality',{}).get('reliable'):
                    expected,mask=expected_in_source(reference,alignment['source_to_reference_homography'],image.shape[:2])
                    native=[native[i] for i in valid_boxes([r['box_xyxy'] for r in native],mask)]
                    if native:
                        if encoder is None:
                            import dino_feature_diff as dino
                            encoder=dino._model();encoder.eval().requires_grad_(False);torch.set_num_threads(1)
                        boxes=[r['box_xyxy'] for r in native];values=paired_features(embeddings(encoder,image,boxes),embeddings(encoder,expected,boxes))
                        with torch.inference_mode():scores=head(values).softmax(dim=1).tolist()
                else:native=[]
            trial=select(current,native,scores,digest);fixed=median/'outer'/(stem+'_predictions.json');pins[str(fixed)]=sha(fixed);assert current==load(fixed)['trial']
            assert trial['all_predictions'][:len(current['all_predictions'])]==current['all_predictions']
            save(folder/(stem+'_predictions.json'),dict(image=name,current=current,trial=trial,proposals=native,probabilities=scores,head_sha256=digest,alignment=alignment))
            targets=read_targets('outer',name,[2736,3648],entry['label_sha256'],pins)
            rows.append(original.measure(name,current,trial,targets))
    return summarize(folder,rows,pins,False)


if __name__=='__main__':main()
