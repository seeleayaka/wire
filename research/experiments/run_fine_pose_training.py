"""Single fixed fine-candidate TRAIN-only grouped hard-negative experiment."""
import copy
import os
import sys
import time
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,REPO,DATA,load,save,sha,read_image,REFERENCE_SHA
from current_port_baseline_audit import BASE,read_targets
from audit_port_multiscale_acceptance import metric,matches
from inspection_agent.paired_native_pose import native_pose_runtime_fingerprint
from inspection_agent.paired_port_features import expected_in_source,valid_boxes,embeddings,paired_features
from paired_pose_native_training import curate,native_label
from raw_pose_consensus import select
from port_semantic_verifier import fit_head
SOURCE=ROOT/'artifacts/fine_native_consensus_20261004/train'
OLD=ROOT/'artifacts/paired_pose_native_training_20261004/training'
OUT=ROOT/'artifacts/fine_pose_training_20261004'
PLAN=ROOT/'artifacts/fine_pose_training_preregistration_20261004/PLAN.md'

def main():
    if OUT.exists():raise FileExistsError('Preserve TRAIN-only head experiment')
    import torch
    import cv2
    import dino_feature_diff as dino
    torch.set_num_threads(2);cv2.setNumThreads(1)
    OUT.mkdir();started=time.monotonic();frozen=native_pose_runtime_fingerprint(REPO)
    groupspath=ROOT/'artifacts/port_training_multiscale_20261002/protocol.json';groups=load(groupspath)
    names=sorted(groups['train_sources']);assert len(names)==192 and set(names).isdisjoint(groups['inner_val_sources'])
    source_folds={name:i%3 for i,name in enumerate(names)}
    referencepath=DATA/'images/train01/normal_073.JPG';assert sha(referencepath)==REFERENCE_SHA
    pins={str(p):sha(p) for p in (Path(__file__),PLAN,OLD/'features.pt',OLD/'samples.json',groupspath,
        Path(__file__).with_name('paired_pose_native_training.py'),Path(__file__).with_name('raw_pose_consensus.py'),
        Path(__file__).with_name('port_semantic_verifier.py'),referencepath,REPO/'inspection_agent/paired_port_features.py')}
    progress=lambda **kw:save(OUT/'progress.json',dict(status='running',pid=os.getpid(),seconds=round(time.monotonic()-started,2),**kw))
    save(OUT/'protocol.json',dict(pins=pins,runtime=frozen,train_only=True,no_validation_fitting=True,
        source_classifier_only_OOF=True,detectors_and_accepted_prefix_not_OOF=True,fixed400_seed0=True,no_deployment=True,field_accuracy=False))
    stages={}
    def finish(status,stage):
        assert all(sha(Path(p))==v for p,v in pins.items()) and native_pose_runtime_fingerprint(REPO)==frozen
        result=dict(status=status,failed_stage=stage,stages=stages,pins=pins,runtime=frozen,
            no_deployment=True,no_validation_fitting=True,classifier_only_OOF=True,field_accuracy=False,seconds=round(time.monotonic()-started,2))
        save(OUT/'report.json',result);save(OUT/'progress.json',dict(status=status,failed_stage=stage,seconds=result['seconds']));print(dict(status=status,stages=stages),flush=True)
    def evaluate(stage,scores,digests,records):
        folder=OUT/stage;folder.mkdir();rows=[];offset=0
        for entry in load(BASE/'train/report.json')['cases']:
            name=entry['image'];case=load(SOURCE/(Path(name).stem+'_predictions.json'));candidates=case['proposals'];local=records[offset:offset+len(candidates)]
            assert all(r['image']==name for r in local)
            values=scores[offset:offset+len(candidates)].tolist();offset+=len(candidates)
            digest=digests[source_folds[name]];current=case['current'];trial=select(current,candidates,values,digest)
            targets=read_targets('train',name,[2736,3648],entry['label_sha256'],pins)
            old=matches(current['all_predictions'],targets)[0];new=matches(trial['all_predictions'],targets)[0]
            rows.append(dict(image=name,current=metric(current['all_predictions'],targets),trial=metric(trial['all_predictions'],targets),gained=sorted(new-old),lost=sorted(old-new)))
            save(folder/(Path(name).stem+'_predictions.json'),dict(image=name,current=current,trial=trial,proposals=candidates,probabilities=values,head_sha256=digest))
        assert offset==len(scores)==len(records)
        totals={version:{k:sum(row[version][k] for row in rows) for k in ('tp','unmatched','fn','predictions','targets')} for version in ('current','trial')}
        normal=sum(row['trial']['predictions'] for row in rows if row['image'].startswith('normal_'))
        qualifies=totals['trial']['tp']>295 and totals['trial']['unmatched']<=4 and not any(row['lost'] for row in rows) and normal==0
        save(folder/'report.json',dict(status='complete',qualifies=qualifies,summary=totals,cases=rows,normal_cues=normal))
        stages[stage]=dict(qualifies=qualifies,summary=totals);print(dict(stage=stage,**stages[stage]),flush=True)
        return qualifies
    try:
        old=torch.load(OLD/'features.pt',map_location='cpu',weights_only=True);oldmeta=load(OLD/'samples.json')
        assert old['features'].shape==(7932,6144) and len(oldmeta)==7932
        assert all(row['label']==int(old['labels'][i]) and row['fold']==int(old['folds'][i]) for i,row in enumerate(oldmeta))
        reference=read_image(referencepath);encoder=None;raw_vectors=[];raw_records=[];self_vectors=[];self_records=[]
        for i,entry in enumerate(load(BASE/'train/report.json')['cases']):
            name=entry['image'];path=SOURCE/(Path(name).stem+'_predictions.json');pins[str(path)]=sha(path);case=load(path);candidates=case['proposals']
            progress(phase='TRAIN_raw_and_reference_self_features',image=name,completed=i,total=192,samples=len(raw_records))
            if not candidates:continue
            source=DATA/'images/train01'/name;pins[str(source)]=sha(source);image=read_image(source)
            assert case['alignment']['alignment_quality']['reliable']
            expected,mask=expected_in_source(reference,case['alignment']['source_to_reference_homography'],image.shape[:2])
            boxes=[p['box_xyxy'] for p in candidates];assert valid_boxes(boxes,mask)==list(range(len(boxes)))
            if encoder is None:encoder=dino._model();encoder.eval().requires_grad_(False);torch.set_num_threads(2)
            observed=embeddings(encoder,image,boxes);expected_vectors=embeddings(encoder,expected,boxes)
            vectors=paired_features(observed,expected_vectors);twins=paired_features(expected_vectors,expected_vectors)
            targets=read_targets('train',name,[2736,3648],entry['label_sha256'],pins)
            for j,p in enumerate(candidates):
                raw_records.append(dict(image=name,fold=source_folds[name],box=p['box_xyxy'],label=native_label(p,targets),kind='raw_native_pose'))
                self_records.append(dict(image=name,fold=source_folds[name],box=p['box_xyxy'],label=0,kind='raw_reference_self',observed_is_expected_reference=True))
            raw_vectors.append(vectors);self_vectors.append(twins)
        raw=torch.cat(raw_vectors);twins=torch.cat(self_vectors);assert raw.shape==twins.shape==(969,6144)
        joined=torch.cat((old['features'],raw,twins));records=oldmeta+raw_records+self_records
        kept,duplicates,conflicts=curate(records,source_folds)
        for a,b in duplicates:assert torch.allclose(joined[a],joined[b],atol=1e-5,rtol=1e-4),(a,b)
        features=joined[kept];labels=torch.tensor([records[i]['label'] for i in kept]);folds=torch.tensor([records[i]['fold'] for i in kept])
        assert torch.isfinite(features).all()
        training=OUT/'training';training.mkdir();torch.save(dict(features=features,labels=labels,folds=folds),training/'features.pt')
        torch.save(dict(features=raw,reference_self_features=twins),training/'raw_features.pt');save(training/'raw_samples.json',raw_records)
        save(training/'samples.json',[records[i] for i in kept]);save(training/'curation.json',dict(input_samples=len(records),retained=len(kept),duplicates=duplicates,conflicts=conflicts,
            raw_label_counts=torch.bincount(torch.tensor([r['label'] for r in raw_records]),minlength=3).tolist()))
        for p in training.iterdir():pins[str(p)]=sha(p)
        heads=OUT/'heads_oof';heads.mkdir();oof=torch.zeros((len(raw),3));rawfolds=torch.tensor([r['fold'] for r in raw_records]);digests={}
        for fold in range(3):
            progress(phase='fixed400_source_group_OOF',fold=fold)
            held=folds==fold
            assert {records[i]['image'] for i in kept if records[i]['fold']==fold}.isdisjoint({records[i]['image'] for i in kept if records[i]['fold']!=fold})
            head=fit_head(features[~held],labels[~held]);path=heads/('fold'+str(fold)+'.pt');torch.save(head.state_dict(),path);digests[fold]=sha(path);pins[str(path)]=digests[fold]
            with torch.inference_mode():oof[rawfolds==fold]=head(raw[rawfolds==fold]).softmax(dim=1)
        if not evaluate('source_train_oof',oof,digests,raw_records):finish('rejected','source_train_oof');return
        progress(phase='fixed400_full_head');head=fit_head(features,labels);folder=OUT/'full';folder.mkdir();path=folder/'last_head.pt'
        torch.save(dict(state_dict=head.state_dict(),input_dimensions=6144,classes=['other','unplugged_plug','unplugged_jack'],encoder_sha256=frozen['encoder']),path);digest=sha(path);pins[str(path)]=digest
        with torch.inference_mode():scores=head(raw).softmax(dim=1)
        if not evaluate('full_train',scores,{i:digest for i in range(3)},raw_records):finish('rejected','full_train');return
        finish('TRAIN_pass_requires_separate_inner_outer_and_actual_gates',None)
    except BaseException as error:
        save(OUT/'progress.json',dict(status='failed',error=type(error).__name__+': '+str(error),seconds=round(time.monotonic()-started,2)));raise

if __name__=='__main__':main()
