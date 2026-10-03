"""Single fixed TRAIN-only, source-grouped core/ring descriptor head trial."""
import os
import sys
import time
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,REPO,DATA,load,save,sha,read_image,REFERENCE_SHA
from current_port_baseline_audit import BASE,read_targets
from inspection_agent.paired_native_pose import native_pose_runtime_fingerprint
from inspection_agent.paired_port_features import expected_in_source
from inspection_agent.paired_native_pose_features import select
from audit_port_multiscale_acceptance import metric,matches
from paired_core_ring import descriptor,normalize_pair,fold_standardization,DIMENSIONS
from port_semantic_verifier import fit_head

TRAINING=ROOT/'artifacts/fine_pose_training_20261004/training'
SOURCE=ROOT/'artifacts/fine_native_consensus_20261004/train'
ALIGNMENTS=ROOT/'artifacts/paired_port_semantics_20261003/features_train'
OUT=ROOT/'artifacts/paired_core_ring_20261004'
PLAN=ROOT/'artifacts/paired_core_ring_preregistration_20261004/PLAN.md'


def main():
    import cv2
    import numpy as np
    import torch
    import psutil
    torch.set_num_threads(2);cv2.setNumThreads(1)
    if OUT.exists():raise FileExistsError('Preserve fixed descriptor trial')
    assert psutil.virtual_memory().available>6*2**30,'Do not compete for inference memory'
    frozen=native_pose_runtime_fingerprint(REPO)
    previous=ROOT/'artifacts/fine_pose_training_20261004/report.json'
    report=load(previous);auditpath=ROOT/'artifacts/fine_pose_training_audit_20261004/report.json'
    audit=load(auditpath)
    assert report['runtime']==frozen and report['status']=='rejected'
    assert audit['status']=='complete' and audit['source_report_sha256']==sha(previous)
    assert all(sha(Path(p))==v for p,v in report['pins'].items())
    groupspath=ROOT/'artifacts/port_training_multiscale_20261002/protocol.json'
    groups=load(groupspath);names=sorted(groups['train_sources']);assert len(names)==192
    assert set(names).isdisjoint(groups['inner_val_sources'])
    source_folds={n:i%3 for i,n in enumerate(names)}
    referencepath=DATA/'images/train01/normal_073.JPG';assert sha(referencepath)==REFERENCE_SHA
    pins={str(p):sha(p) for p in (Path(__file__),Path(__file__).with_name('paired_core_ring.py'),
        Path(__file__).with_name('robust_port_exposure.py'),Path(__file__).with_name('port_semantic_verifier.py'),
        PLAN,previous,auditpath,groupspath,referencepath,BASE/'train/report.json',
        TRAINING/'features.pt',TRAINING/'samples.json',TRAINING/'raw_features.pt',TRAINING/'raw_samples.json',
        REPO/'inspection_agent/paired_port_features.py',REPO/'inspection_agent/paired_native_pose_features.py')}
    data=torch.load(TRAINING/'features.pt',map_location='cpu',weights_only=True)
    raw=torch.load(TRAINING/'raw_features.pt',map_location='cpu',weights_only=True)['features']
    samples=load(TRAINING/'samples.json');raw_records=load(TRAINING/'raw_samples.json')
    features,labels,folds=data['features'],data['labels'],data['folds']
    assert features.shape==(9808,6144) and raw.shape==(969,6144)
    assert len(samples)==9808 and len(raw_records)==969
    assert all(r['fold']==source_folds[r['image']]==int(folds[i]) and r['label']==int(labels[i]) for i,r in enumerate(samples))
    assert all(r['fold']==source_folds[r['image']] for r in raw_records)
    by_source={n:[] for n in names};raw_by_source={n:[] for n in names}
    for i,row in enumerate(samples):by_source[row['image']].append(i)
    for i,row in enumerate(raw_records):raw_by_source[row['image']].append(i)
    OUT.mkdir();started=time.monotonic();stages={}
    progress=lambda **kw:save(OUT/'progress.json',dict(status='running',pid=os.getpid(),seconds=round(time.monotonic()-started,2),**kw))
    save(OUT/'protocol.json',dict(pins=pins,runtime=frozen,train_sources=names,dimensions=6144+DIMENSIONS,
        descriptor_dimensions=DIMENSIONS,source_group_folds=source_folds,seed=0,steps=400,
        learning_rate=.01,weight_decay=.001,standardization_train_only=True,
        no_validation_fitting=True,no_deployment=True,field_accuracy=False))
    def evaluate(stage,scores,digests):
        folder=OUT/stage;folder.mkdir();rows=[];offset=0
        for entry in load(BASE/'train/report.json')['cases']:
            name=entry['image'];path=SOURCE/(Path(name).stem+'_predictions.json');pins[str(path)]=sha(path);case=load(path)
            n=len(case['proposals']);local=raw_records[offset:offset+n]
            assert len(local)==n and all(r['image']==name and r['box']==p['box_xyxy'] for r,p in zip(local,case['proposals']))
            values=scores[offset:offset+n].tolist();offset+=n
            digest=digests[source_folds[name]];current=case['current'];trial=select(current,case['proposals'],values,digest)
            targets=read_targets('train',name,[2736,3648],entry['label_sha256'],pins)
            old=matches(current['all_predictions'],targets)[0];new=matches(trial['all_predictions'],targets)[0]
            rows.append(dict(image=name,current=metric(current['all_predictions'],targets),trial=metric(trial['all_predictions'],targets),gained=sorted(new-old),lost=sorted(old-new)))
            save(folder/(Path(name).stem+'_predictions.json'),dict(image=name,current=current,trial=trial,
                proposals=case['proposals'],probabilities=values,head_sha256=digest))
        assert offset==969
        totals={v:{k:sum(r[v][k] for r in rows) for k in ('tp','unmatched','fn','predictions','targets')} for v in ('current','trial')}
        normal=sum(r['trial']['predictions'] for r in rows if r['image'].startswith('normal_'))
        assert totals['current']['tp']==295 and totals['current']['unmatched']==4
        qualifies=totals['trial']['tp']>295 and totals['trial']['unmatched']<=4 and normal==0 and not any(r['lost'] for r in rows)
        summary=dict(qualifies=qualifies,summary=totals,normal_cues=normal)
        save(folder/'report.json',dict(status='complete',**summary,cases=rows));stages[stage]=summary
        print(dict(stage=stage,**summary),flush=True)
        return qualifies
    def finish(failed):
        assert all(sha(Path(p))==v for p,v in pins.items()) and native_pose_runtime_fingerprint(REPO)==frozen
        result=dict(status='rejected' if failed else 'TRAIN_pass_requires_fresh_holdouts_and_actual_gates',
            failed_stage=failed,stages=stages,pins=pins,runtime=frozen,classifier_only_source_OOF=True,
            no_validation_fitting=True,no_deployment=True,field_accuracy=False,seconds=round(time.monotonic()-started,2))
        save(OUT/'report.json',result);save(OUT/'progress.json',dict(status=result['status'],seconds=result['seconds']))
        print(dict(status=result['status'],stages=stages,seconds=result['seconds']),flush=True)
    try:
        reference=read_image(referencepath);extra=np.zeros((9808,DIMENSIONS),np.float32);raw_extra=np.zeros((969,DIMENSIONS),np.float32)
        records=[]
        for completed,name in enumerate(names):
            progress(phase='TRAIN_registered_body_ring_descriptors',image=name,completed=completed,total=192)
            source=DATA/'images/train01'/name;pins[str(source)]=sha(source)
            indices=by_source[name];ri=raw_by_source[name]
            if not indices and not ri:
                records.append(dict(image=name,status='no_existing_valid_training_examples',samples=0,raw=0));continue
            alignmentpath=ALIGNMENTS/(Path(name).stem+'_source.json');pins[str(alignmentpath)]=sha(alignmentpath)
            aligned=load(alignmentpath);assert aligned['source_sha256']==pins[str(source)]
            assert aligned['alignment']['alignment_quality']['reliable']
            image=read_image(source);expected,mask=expected_in_source(reference,aligned['alignment']['source_to_reference_homography'],image.shape[:2])
            normalized,metadata=normalize_pair(image,expected,mask);cache={}
            def extract(row,self_pair):
                key=(self_pair,*row['box'])
                if key not in cache:cache[key]=descriptor(expected if self_pair else normalized,expected,mask,row['box'])
                return cache[key]
            for i in indices:
                row=samples[i];is_self=row.get('observed_is_expected_reference',False) or row['kind'] in ('reference_self','jitter_reference_self')
                if is_self:assert torch.allclose(features[i,:1536],features[i,1536:3072],atol=1e-6,rtol=1e-5) and row['label']==0
                extra[i]=extract(row,is_self)
            for i in ri:raw_extra[i]=extract(raw_records[i],False)
            records.append(dict(image=name,status='complete',samples=len(indices),raw=len(ri),
                source_sha256=pins[str(source)],alignment_path=str(alignmentpath),alignment_sha256=pins[str(alignmentpath)],
                normalization=metadata,self_pairs=sum(samples[i].get('observed_is_expected_reference',False) or samples[i]['kind'] in ('reference_self','jitter_reference_self') for i in indices)))
        assert np.isfinite(extra).all() and np.isfinite(raw_extra).all()
        descriptorpath=OUT/'descriptors.pt';torch.save(dict(training=torch.from_numpy(extra),raw=torch.from_numpy(raw_extra)),descriptorpath)
        save(OUT/'descriptor_sources.json',records);pins[str(descriptorpath)]=sha(descriptorpath);pins[str(OUT/'descriptor_sources.json')]=sha(OUT/'descriptor_sources.json')
        extra=torch.from_numpy(extra);raw_extra=torch.from_numpy(raw_extra);rawfolds=torch.tensor([r['fold'] for r in raw_records])
        heads=OUT/'heads_oof';heads.mkdir();oof=torch.zeros((969,3));digests={}
        for fold in range(3):
            progress(phase='fixed400_core_ring_source_OOF',fold=fold)
            held=folds==fold;rawheld=rawfolds==fold
            assert {r['image'] for r in samples if r['fold']==fold}.isdisjoint({r['image'] for r in samples if r['fold']!=fold})
            train_extra,eval_extra,mean,std=fold_standardization(extra[~held],raw_extra[rawheld])
            head=fit_head(torch.cat((features[~held],train_extra),1),labels[~held]);path=heads/('fold'+str(fold)+'.pt')
            torch.save(dict(state_dict=head.state_dict(),mean=mean,std=std),path);digests[fold]=sha(path);pins[str(path)]=digests[fold]
            with torch.inference_mode():oof[rawheld]=head(torch.cat((raw[rawheld],eval_extra),1)).softmax(1)
        if not evaluate('source_train_oof',oof,digests):finish('source_train_oof');return
        progress(phase='fixed400_core_ring_full_head');train_extra,eval_extra,mean,std=fold_standardization(extra,raw_extra)
        head=fit_head(torch.cat((features,train_extra),1),labels);full=OUT/'full';full.mkdir();path=full/'last_head.pt'
        torch.save(dict(state_dict=head.state_dict(),mean=mean,std=std,dimensions=6144+DIMENSIONS,
            classes=['other','unplugged_plug','unplugged_jack'],encoder_sha256=frozen['encoder']),path)
        digest=sha(path);pins[str(path)]=digest
        with torch.inference_mode():scores=head(torch.cat((raw,eval_extra),1)).softmax(1)
        if not evaluate('full_train',scores,{i:digest for i in range(3)}):finish('full_train');return
        finish(None)
    except BaseException as error:
        save(OUT/'progress.json',dict(status='failed',error=type(error).__name__+': '+str(error),seconds=round(time.monotonic()-started,2)));raise


if __name__=='__main__':main()
