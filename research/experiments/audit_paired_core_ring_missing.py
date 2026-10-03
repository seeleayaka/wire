"""Re-extract all paired pixels; independently check grouped scaling/scores/GT."""
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
from paired_core_ring_missing import descriptor,normalize_pair,DIMENSIONS

SOURCE=ROOT/'artifacts/paired_core_ring_missing_20261004'
OUT=ROOT/'artifacts/paired_core_ring_missing_replay_20261004'
TRAINING=ROOT/'artifacts/fine_pose_training_20261004/training'
FINE=ROOT/'artifacts/fine_native_consensus_20261004/train'
ALIGNMENTS=ROOT/'artifacts/paired_port_semantics_20261003/features_train'


def main():
    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(2);cv2.setNumThreads(1)
    if OUT.exists():raise FileExistsError('Preserve all-source descriptor audit')
    report=load(SOURCE/'report.json');digest=sha(SOURCE/'report.json');pins=report['pins']
    assert report['status'] in ('rejected','TRAIN_pass_requires_fresh_holdouts_and_actual_gates')
    assert all(sha(Path(p))==v for p,v in pins.items())
    assert native_pose_runtime_fingerprint(REPO)==report['runtime']
    protocol=load(SOURCE/'protocol.json');names=protocol['train_sources'];foldmap={n:i%3 for i,n in enumerate(names)}
    assert len(names)==192 and names==sorted(names) and foldmap=={n:int(f) for n,f in protocol['source_group_folds'].items()}
    data=torch.load(TRAINING/'features.pt',map_location='cpu',weights_only=True)
    raw=torch.load(TRAINING/'raw_features.pt',map_location='cpu',weights_only=True)['features']
    extra=torch.load(SOURCE/'descriptors.pt',map_location='cpu',weights_only=True)
    samples=load(TRAINING/'samples.json');rawrecords=load(TRAINING/'raw_samples.json')
    assert extra['training'].shape==(9808,DIMENSIONS) and extra['raw'].shape==(969,DIMENSIONS)
    assert all(r['fold']==foldmap[r['image']]==int(data['folds'][i]) and r['label']==int(data['labels'][i]) for i,r in enumerate(samples))
    assert all(r['fold']==foldmap[r['image']] for r in rawrecords)
    by_source={n:[] for n in names};raw_by_source={n:[] for n in names}
    for i,row in enumerate(samples):by_source[row['image']].append(i)
    for i,row in enumerate(rawrecords):raw_by_source[row['image']].append(i)
    source_records=load(SOURCE/'descriptor_sources.json');assert [r['image'] for r in source_records]==names
    referencepath=DATA/'images/train01/normal_073.JPG';assert sha(referencepath)==REFERENCE_SHA
    reference=read_image(referencepath);OUT.mkdir();started=time.monotonic();checked=0
    for index,name in enumerate(names):
        save(OUT/'progress.json',dict(status='running',phase='all_source_pixel_descriptor_replay',completed=index,total=192,seconds=round(time.monotonic()-started,2)))
        source=DATA/'images/train01'/name;assert sha(source)==pins[str(source)]
        indices=by_source[name];ri=raw_by_source[name];record=source_records[index]
        assert record['samples']==len(indices) and record['raw']==len(ri)
        if not indices and not ri:assert record['status']=='no_existing_valid_training_examples';continue
        path=ALIGNMENTS/(Path(name).stem+'_source.json');assert str(path)==record['alignment_path'] and sha(path)==record['alignment_sha256']==pins[str(path)]
        aligned=load(path);assert aligned['source_sha256']==pins[str(source)] and aligned['alignment']['alignment_quality']['reliable']
        fine=load(FINE/(Path(name).stem+'_predictions.json'))
        if fine['alignment']:assert fine['alignment']['source_to_reference_homography']==aligned['alignment']['source_to_reference_homography']
        image=read_image(source);expected,mask=expected_in_source(reference,aligned['alignment']['source_to_reference_homography'],image.shape[:2])
        normalized,metadata=normalize_pair(image,expected,mask);assert metadata==record['normalization']
        cache={}
        def extract(row,self_pair):
            key=(self_pair,*row['box'])
            if key not in cache:cache[key]=descriptor(expected if self_pair else normalized,expected,mask,row['box'])
            return cache[key]
        self_count=0
        for i in indices:
            row=samples[i];is_self=row.get('observed_is_expected_reference',False) or row['kind'] in ('reference_self','jitter_reference_self')
            if is_self:
                self_count+=1;assert row['label']==0
                assert torch.allclose(data['features'][i,:1536],data['features'][i,1536:3072],atol=1e-6,rtol=1e-5)
            np.testing.assert_allclose(extract(row,is_self),extra['training'][i].numpy(),atol=1e-6,rtol=1e-6);checked+=1
        for i in ri:np.testing.assert_allclose(extract(rawrecords[i],False),extra['raw'][i].numpy(),atol=1e-6,rtol=1e-6);checked+=1
        assert self_count==record['self_pairs']
    all_scores={}
    rawfolds=torch.tensor([r['fold'] for r in rawrecords])
    for stage in report['stages']:
        scores=torch.zeros((969,3))
        for fold in range(3) if stage=='source_train_oof' else [None]:
            mask=data['folds']!=fold if fold is not None else torch.ones(len(samples),dtype=torch.bool)
            selected=rawfolds==fold if fold is not None else torch.ones(len(rawrecords),dtype=torch.bool)
            values=extra['training'][mask]
            # Independent population-moment formula; do not call training scaling helper.
            mean=values.double().mean(0).float()
            std=((values.double()-mean.double()).square().mean(0).sqrt()).float().clamp_min(.01)
            path=SOURCE/'heads_oof'/('fold'+str(fold)+'.pt') if fold is not None else SOURCE/'full/last_head.pt'
            saved=torch.load(path,map_location='cpu',weights_only=True);assert sha(path)==pins[str(path)]
            torch.testing.assert_close(mean,saved['mean'],atol=2e-6,rtol=1e-5)
            torch.testing.assert_close(std,saved['std'],atol=2e-6,rtol=1e-5)
            joined=torch.cat((raw[selected],((extra['raw'][selected]-saved['mean'])/saved['std']).clamp(-5,5)),1)
            state=saved['state_dict'];assert state['weight'].shape==(3,6191)
            with torch.inference_mode():scores[selected]=F.linear(joined,state['weight'],state['bias']).softmax(1)
        # Recompute compulsory missing-input abstention independently.
        presence=extra['raw'][:,-1]
        assert ((presence==0)|(presence==1)).all()
        assert (extra['raw'][presence==0]==0).all()
        scores[presence==0]=torch.tensor([1.,0.,0.])
        totals={v:{k:0 for k in ('tp','unmatched','fn','predictions','targets')} for v in ('current','trial')}
        normal=0;lost=[];offset=0
        for entry in load(BASE/'train/report.json')['cases']:
            name=entry['image'];path=SOURCE/stage/(Path(name).stem+'_predictions.json');saved=load(path)
            original=load(FINE/(Path(name).stem+'_predictions.json'));n=len(original['proposals'])
            assert saved['current']==original['current'] and saved['proposals']==original['proposals']
            np.testing.assert_allclose(scores[offset:offset+n].numpy(),np.array(saved['probabilities']).reshape(n,3),atol=1e-6,rtol=1e-5);offset+=n
            headpath=SOURCE/'heads_oof'/('fold'+str(foldmap[name])+'.pt') if stage=='source_train_oof' else SOURCE/'full/last_head.pt'
            assert saved['head_sha256']==sha(headpath)
            trial=select(saved['current'],saved['proposals'],saved['probabilities'],saved['head_sha256']);assert trial==saved['trial']
            labelpins={};targets=read_targets('train',name,[2736,3648],entry['label_sha256'],labelpins)
            assert all(pins[p]==v for p,v in labelpins.items())
            lost.extend(matches(saved['current']['all_predictions'],targets)[0]-matches(trial['all_predictions'],targets)[0])
            for version in ('current','trial'):
                score=metric(saved[version]['all_predictions'],targets)
                for key in totals[version]:totals[version][key]+=score[key]
            if name.startswith('normal_'):normal+=len(trial['all_predictions'])
        assert offset==969 and totals==report['stages'][stage]['summary']
        qualifies=totals['trial']['tp']>295 and totals['trial']['unmatched']<=4 and not lost and normal==0
        assert qualifies==report['stages'][stage]['qualifies'] and normal==report['stages'][stage]['normal_cues']
        all_scores[stage]=dict(qualifies=qualifies,summary=totals)
    assert all(sha(Path(p))==v for p,v in pins.items()) and sha(SOURCE/'report.json')==digest
    assert native_pose_runtime_fingerprint(REPO)==report['runtime']
    result=dict(status='pass',experiment_status=report['status'],sources=192,training_examples=9808,raw_proposals=969,
        pixel_descriptors_recomputed=checked,descriptor_implementation_shared=True,
        missing_training_examples=int((extra['training'][:,-1]==0).sum()),missing_raw_candidates=int((extra['raw'][:,-1]==0).sum()),missing_candidates_force_other_replayed=True,
        independent_grouped_scaling_head_score_and_selector_GT_replay=True,stages=all_scores,
        source_report_sha256=digest,auditor_sha256=sha(Path(__file__)),no_deployment=True,field_accuracy=False,
        seconds=round(time.monotonic()-started,2))
    save(OUT/'report.json',result);save(OUT/'progress.json',dict(status='pass',seconds=result['seconds']));print(result,flush=True)


if __name__=='__main__':main()


