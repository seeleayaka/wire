"""Finite TRAIN-only native hard-negative classifier, unchanged runtime."""
import copy
import os
import sys
import time
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,REPO,DATA,load,save,sha
from current_port_baseline_audit import BASE,read_targets
from audit_port_multiscale_acceptance import metric,matches
from port_semantic_verifier import fit_head
from paired_pose_native_training import curate,native_label
from paired_pose_head_agreement import select
from inspection_agent.paired_port_geometry import HEAD_SHA
from inspection_agent.paired_median_geometry import median_runtime_fingerprint
OLD=ROOT/'artifacts/paired_positive_jitter_20261003/features_train'
NATIVE=ROOT/'artifacts/paired_pose_jitter_agreement_20261003'
INPUTS=ROOT/'artifacts/paired_pose_search_inputs_20261003'
MEDIAN=ROOT/'artifacts/paired_median_current_head_20261003'
OUT=ROOT/'artifacts/paired_pose_native_training_20261004'
PLAN=ROOT/'artifacts/paired_pose_native_training_preregistration_20261004/PLAN.md'


def stage_report(folder,rows,pins,strict_gain):
    totals={v:{k:sum(r[v][k] for r in rows) for k in ('tp','unmatched','fn','predictions','targets')} for v in ('current','trial')}
    gain=totals['trial']['tp']>totals['current']['tp'] if strict_gain else totals['trial']['tp']>=totals['current']['tp']
    normal=sum(r['trial']['predictions'] for r in rows if r['image'].startswith('normal_'))
    qualifies=gain and totals['trial']['unmatched']<=totals['current']['unmatched'] and not any(r['lost'] for r in rows) and normal==0
    assert {p:sha(Path(p)) for p in pins}==pins
    result=dict(status='complete',qualifies=qualifies,summary=totals,cases=rows,normal_cues=normal)
    save(folder/'report.json',result);print(str(dict(stage=folder.name,qualifies=qualifies,summary=totals)),flush=True)
    return result


def measure(name,current,trial,targets):
    old,new=matches(current['all_predictions'],targets)[0],matches(trial['all_predictions'],targets)[0]
    return dict(image=name,current=metric(current['all_predictions'],targets),trial=metric(trial['all_predictions'],targets),gained=sorted(new-old),lost=sorted(old-new))


def evaluate_train(folder,metadata,scores,digests,pins):
    folder.mkdir();rows=[];offset=0
    for entry in load(BASE/'train/report.json')['cases']:
        name=entry['image'];path=NATIVE/(Path(name).stem+'_predictions.json');pins[str(path)]=sha(path);case=load(path)
        candidates=case['proposals'];local=metadata[offset:offset+len(candidates)]
        assert all(r['image']==name for r in local) and [r['proposal'] for r in local]==candidates
        digest=digests[local[0]['fold']] if local else digests[0]
        values=scores[offset:offset+len(candidates)].tolist();offset+=len(candidates)
        current=case['current'];trial=select(current,candidates,case['original_probabilities'],values,HEAD_SHA,digest)
        assert trial['all_predictions'][:len(current['all_predictions'])]==current['all_predictions']
        save(folder/path.name,dict(image=name,current=current,trial=trial,proposals=candidates,
            original_probabilities=case['original_probabilities'],auxiliary_probabilities=values,auxiliary_head_sha256=digest))
        targets=read_targets('train',name,[2736,3648],entry['label_sha256'],pins)
        rows.append(measure(name,current,trial,targets))
    assert offset==len(metadata)==len(scores)
    return stage_report(folder,rows,pins,True)


def main():
    if OUT.exists():raise FileExistsError('Preserve native hard-negative experiment')
    import torch
    import cv2
    torch.set_num_threads(2);cv2.setNumThreads(1)
    sources=load(ROOT/'artifacts/port_training_multiscale_20261002/protocol.json')
    names=sorted(sources['train_sources']);assert len(names)==192
    assert set(names).isdisjoint(sources['inner_val_sources'])
    source_folds={name:i%3 for i,name in enumerate(names)}
    prior=load(NATIVE/'report.json');assert prior['status']=='complete' and prior['native_pairs']==679
    old_report=load(OLD/'report.json');assert old_report['status']=='complete'
    pins=dict(prior['pins']);pins.update(old_report['pins'])
    for path in (Path(__file__),Path(__file__).with_name('paired_pose_native_training.py'),
        Path(__file__).with_name('paired_pose_head_agreement.py'),Path(__file__).with_name('paired_pose_search.py'),
        Path(__file__).with_name('port_semantic_verifier.py'),PLAN,OLD/'report.json',OLD/'features.pt',OLD/'samples.json',
        NATIVE/'report.json',NATIVE/'native_features.pt',NATIVE/'native_samples.json'):
        pins[str(path)]=sha(path)
    assert pins[str(OLD/'features.pt')]==old_report['aggregate_feature_sha256']
    assert pins[str(OLD/'samples.json')]==old_report['samples_sha256']
    assert pins[str(NATIVE/'native_features.pt')]=='6449c09d16cde5c232741a9db331d934691ebb190bd54b324bd5a51798f96bb4'
    assert pins[str(NATIVE/'native_samples.json')]=='3d575d8997c695dacef7bac742ab9dbf8bf92a1a6bfa88b718bc7aa97f65848b'
    assert {p:sha(Path(p)) for p in pins}==pins
    frozen=median_runtime_fingerprint(REPO);OUT.mkdir();started=time.monotonic();stages={}
    save(OUT/'protocol.json',dict(pins=pins,median_runtime_fingerprint=frozen,train_sources=names,
        native_training_GT_only=True,source_classifier_folds=True,YOLO_not_OOF=True,fixed400_steps=True,
        unchanged_p98=True,original_and_auxiliary_same_native=True,no_validation_training=True,no_deployment=True,field_accuracy=False))
    def progress(**kw):save(OUT/'progress.json',dict(status='running',pid=os.getpid(),seconds=round(time.monotonic()-started,2),**kw))
    def finish(status,stage):
        assert {p:sha(Path(p)) for p in pins}==pins and median_runtime_fingerprint(REPO)==frozen
        result=dict(status=status,stage=stage,stages=stages,pins=pins,median_runtime_fingerprint=frozen,
            seconds=round(time.monotonic()-started,2),no_deployment=True,field_accuracy=False)
        save(OUT/'report.json',result);save(OUT/'progress.json',dict(status=status,stage=stage,seconds=result['seconds']))
    try:
        progress(phase='curate_training_sources')
        old=torch.load(OLD/'features.pt',map_location='cpu',weights_only=True);oldmeta=load(OLD/'samples.json')
        native=torch.load(NATIVE/'native_features.pt',map_location='cpu',weights_only=True)['features'];meta=load(NATIVE/'native_samples.json')
        assert old['features'].shape==(7262,6144) and native.shape==(679,6144)
        assert len(oldmeta)==len(old['features']) and len(meta)==len(native)
        assert all(r['label']==int(old['labels'][i]) and r['fold']==int(old['folds'][i]) for i,r in enumerate(oldmeta))
        targets_by_source={entry['image']:read_targets('train',entry['image'],[2736,3648],entry['label_sha256'],pins)
            for entry in load(BASE/'train/report.json')['cases']}
        added=[dict(image=r['image'],fold=r['fold'],box=r['proposal']['box_xyxy'],kind='native_searched_pose',
            label=native_label(r['proposal'],targets_by_source[r['image']])) for r in meta]
        records=copy.deepcopy(oldmeta)+added;joined=torch.cat((old['features'],native))
        kept,duplicates,conflicts=curate(records,source_folds)
        for first,other in duplicates:
            assert torch.allclose(joined[first],joined[other],atol=1e-5,rtol=1e-4),(first,other)
        features=joined[kept];labels=torch.tensor([records[i]['label'] for i in kept]);folds=torch.tensor([records[i]['fold'] for i in kept])
        assert torch.isfinite(features).all()
        prepared=OUT/'training';prepared.mkdir()
        torch.save(dict(features=features,labels=labels,folds=folds),prepared/'features.pt')
        save(prepared/'samples.json',[records[i] for i in kept]);save(prepared/'curation.json',dict(
            input_samples=len(records),retained=len(kept),duplicates=duplicates,conflicts=conflicts,
            native_label_counts=torch.bincount(torch.tensor([r['label'] for r in added]),minlength=3).tolist(),
            retained_label_counts=torch.bincount(labels,minlength=3).tolist()))
        for path in (prepared/'features.pt',prepared/'samples.json',prepared/'curation.json'):pins[str(path)]=sha(path)
        heads=OUT/'heads_oof';heads.mkdir();oof=torch.zeros((len(native),3));digests={}
        native_folds=torch.tensor([r['fold'] for r in meta]);fold_reports=[]
        for fold in range(3):
            progress(phase='fixed400_source_OOF',fold=fold,total=3)
            held=folds==fold;assert {records[i]['image'] for i in kept if records[i]['fold']==fold}.isdisjoint(
                {records[i]['image'] for i in kept if records[i]['fold']!=fold})
            head=fit_head(features[~held],labels[~held]);path=heads/f'fold{fold}_head.pt'
            torch.save(head.state_dict(),path);digests[fold]=sha(path);pins[str(path)]=digests[fold]
            selected=native_folds==fold
            with torch.inference_mode():oof[selected]=head(native[selected]).softmax(dim=1)
            fold_reports.append(dict(fold=fold,training_examples=int((~held).sum()),held_examples=int(held.sum()),native_held_examples=int(selected.sum())))
        torch.save(oof,heads/'native_oof_probabilities.pt');pins[str(heads/'native_oof_probabilities.pt')]=sha(heads/'native_oof_probabilities.pt')
        save(heads/'report.json',dict(folds=fold_reports,head_pins=digests,all_source_twins_excluded=True))
        result=evaluate_train(OUT/'source_train_oof',meta,oof,digests,pins)
        stages['source_train_oof']=dict(qualifies=result['qualifies'],summary=result['summary'])
        if not result['qualifies']:finish('rejected','source_train_oof');return
        progress(phase='fixed400_full_head')
        head=fit_head(features,labels);folder=OUT/'full';folder.mkdir();path=folder/'last_head.pt'
        torch.save(dict(state_dict=head.state_dict(),input_dimensions=6144,classes=['other','unplugged_plug','unplugged_jack'],
            encoder_sha256='b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9'),path)
        digest=sha(path);pins[str(path)]=digest
        with torch.inference_mode():scores=head(native).softmax(dim=1)
        result=evaluate_train(OUT/'full_train',meta,scores,{i:digest for i in range(3)},pins)
        stages['full_train']=dict(qualifies=result['qualifies'],summary=result['summary'])
        if not result['qualifies']:finish('rejected','full_train');return
        evaluate_holdouts(head,digest,pins,stages,progress)
        failed=next((stage for stage in ('inner','outer') if stage in stages and not stages[stage]['qualifies']),None)
        finish('rejected' if failed else 'source_pass_requires_reference_ROI_SAM_Qt',failed)
    except BaseException as error:
        save(OUT/'progress.json',dict(status='failed',error=type(error).__name__+': '+str(error)));raise


def evaluate_holdouts(head,digest,pins,stages,progress):
    import torch
    import cv2
    from prepare_paired_port_semantics import read_image,cached_alignments,REFERENCE_SHA
    from paired_port_semantics import expected_in_source,valid_boxes,paired_features,CachedReferenceSIFT
    from port_semantic_verifier import embeddings
    import assembly_auto_review_robust_v3 as registration
    oldpath=REPO/'output/paired_port_geometry_20261003/last_head.pt';assert sha(oldpath)==HEAD_SHA;pins[str(oldpath)]=HEAD_SHA
    oldhead=torch.nn.Linear(6144,3);oldhead.load_state_dict(torch.load(oldpath,map_location='cpu',weights_only=True)['state_dict']);oldhead.eval()
    referencepath=DATA/'images/train01/normal_073.JPG';assert sha(referencepath)==REFERENCE_SHA
    pins[str(referencepath)]=REFERENCE_SHA;reference=read_image(referencepath);alignments=cached_alignments();encoder=None
    with CachedReferenceSIFT(reference):
        for stage,count in (('inner',48),('outer',30)):
            folder=OUT/stage;folder.mkdir();rows=[];entries=load(BASE/stage/'report.json')['cases'];assert len(entries)==count
            for index,entry in enumerate(entries):
                name=entry['image'];stem=Path(name).stem;path=INPUTS/(stage+'_'+stem+'_proposals.json')
                pins[str(path)]=sha(path);case=load(path);current=case['current'];native=case['extended_candidates'];scores=[];original=[];alignment=None
                progress(stage=stage,image=name,completed=index,total=count,phase='fresh_native_holdout_features')
                if native:
                    source=DATA/'images'/('val01' if stage=='outer' else 'train01')/name;pins[str(source)]=sha(source);image=read_image(source)
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
                            with torch.inference_mode():scores=head(values).softmax(dim=1).tolist();original=oldhead(values).softmax(dim=1).tolist()
                    else:native=[]
                trial=select(current,native,original,scores,HEAD_SHA,digest)
                assert trial['all_predictions'][:len(current['all_predictions'])]==current['all_predictions']
                fixed=MEDIAN/stage/(stem+'_predictions.json');pins[str(fixed)]=sha(fixed);assert current==load(fixed)['trial']
                save(folder/(stem+'_predictions.json'),dict(image=name,current=current,trial=trial,proposals=native,
                    original_probabilities=original,auxiliary_probabilities=scores,auxiliary_head_sha256=digest,alignment=alignment))
                targets=read_targets(stage,name,[2736,3648],entry['label_sha256'],pins)
                rows.append(measure(name,current,trial,targets))
            result=stage_report(folder,rows,pins,stage=='inner');stages[stage]=dict(qualifies=result['qualifies'],summary=result['summary'])
            if not result['qualifies']:return


if __name__=='__main__':main()
