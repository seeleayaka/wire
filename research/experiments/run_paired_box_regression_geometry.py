"""TRAIN-only source-excluded relative-regression proposal geometry, no accuracy claim."""
import copy
import os
import sys
import time
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,REPO,DATA,load,save,sha,read_image,REFERENCE_SHA
from current_port_baseline_audit import BASE,read_targets
from inspection_agent.paired_native_pose import native_pose_runtime_fingerprint
from inspection_agent.paired_port_features import expected_in_source,valid_boxes
from audit_port_multiscale_acceptance import overlap,matches
from relative_port_box import encode,decode,bounded,fit
TRAINING=ROOT/'artifacts/fine_pose_training_20261004/training'
NATIVE=ROOT/'artifacts/paired_pose_jitter_agreement_20261003'
CURRENT=ROOT/'artifacts/paired_pose_native_three_20261004/full_train'
PREP=ROOT/'artifacts/paired_support_graph_20261003/source_selections/train'
OUT=ROOT/'artifacts/paired_box_regression_geometry_20261004'
PLAN=ROOT/'artifacts/paired_box_regression_preregistration_20261004/PLAN.md'


def main():
    import cv2
    import torch
    import psutil
    torch.set_num_threads(2);cv2.setNumThreads(1)
    if OUT.exists():raise FileExistsError('Preserve relative geometry trial')
    assert psutil.virtual_memory().available>6*2**30,'No competing memory-heavy job'
    frozen=native_pose_runtime_fingerprint(REPO)
    prior=load(ROOT/'artifacts/fine_pose_training_20261004/report.json')
    assert prior['runtime']==frozen and prior['status']=='rejected'
    audit=load(ROOT/'artifacts/fine_pose_training_audit_20261004/report.json')
    assert audit['status']=='complete' and audit['source_report_sha256']==sha(ROOT/'artifacts/fine_pose_training_20261004/report.json')
    assert all(sha(Path(p))==v for p,v in prior['pins'].items())
    nativeprior=load(ROOT/'artifacts/paired_pose_native_training_20261004/report.json')
    for p in (NATIVE/'native_features.pt',NATIVE/'native_samples.json'):
        assert sha(p)==nativeprior['pins'][str(p)]
    groupspath=ROOT/'artifacts/port_training_multiscale_20261002/protocol.json'
    names=sorted(load(groupspath)['train_sources']);assert len(names)==192
    folds={name:i%3 for i,name in enumerate(names)}
    pins={str(p):sha(p) for p in (Path(__file__),Path(__file__).with_name('relative_port_box.py'),PLAN,
        TRAINING/'features.pt',TRAINING/'samples.json',NATIVE/'native_features.pt',NATIVE/'native_samples.json',
        groupspath,PREP/'index.json',ROOT/'artifacts/fine_pose_training_audit_20261004/report.json')}
    data=torch.load(TRAINING/'features.pt',map_location='cpu',weights_only=True)
    samples=load(TRAINING/'samples.json');assert data['features'].shape==(9808,6144)
    assert all(r['fold']==folds[r['image']]==int(data['folds'][i]) and r['label']==int(data['labels'][i]) for i,r in enumerate(samples))
    native=torch.load(NATIVE/'native_features.pt',map_location='cpu',weights_only=True)['features']
    native_samples=load(NATIVE/'native_samples.json');assert native.shape==(len(native_samples),6144)
    assert all(r['fold']==folds[r['image']] for r in native_samples)
    feature_index={(r['image'],r['proposal']['class_id'],*r['proposal']['box_xyxy']):i for i,r in enumerate(native_samples)}
    assert len(feature_index)==len(native_samples)
    entries={r['image']:r for r in load(BASE/'train/report.json')['cases']}
    targets={name:read_targets('train',name,[2736,3648],entries[name]['label_sha256'],pins) for name in names}
    indices=[];training_records=[];encoded=[]
    for i,row in enumerate(samples):
        if row['label']==0:continue
        closest=max(((overlap(row['box'],t['box']),j,t) for j,t in enumerate(targets[row['image']]) if t['class_id']==row['label']-1),default=None)
        assert closest and closest[0]>=.5,(i,row['image'])
        delta=bounded(encode(row['box'],closest[2]['box']))
        indices.append(i);encoded.append(delta)
        training_records.append(dict(image=row['image'],fold=row['fold'],sample_index=i,
            label=row['label'],target_index=closest[1],box=row['box'],target_box=closest[2]['box'],
            initial_iou=closest[0],relative_target=delta,kind=row['kind']))
    features=data['features'][indices];regression_targets=torch.tensor(encoded,dtype=torch.float32)
    regression_folds=torch.tensor([r['fold'] for r in training_records])
    del data
    OUT.mkdir();started=time.monotonic();heads={};head_sha={};positive_predictions=torch.zeros_like(regression_targets)
    native_predictions=torch.zeros((len(native_samples),4));native_folds=torch.tensor([r['fold'] for r in native_samples])
    save(OUT/'protocol.json',dict(pins=pins,runtime=frozen,source_folds=folds,steps=400,
        seed=0,learning_rate=.01,weight_decay=.001,loss='SmoothL1',head_dimensions=[6144,4],
        no_classifier_change=True,no_validation_read=True,proposal_potential_only=True,
        no_full_head=True,no_deployment=True,field_accuracy=False))
    save(OUT/'regression_training_records.json',training_records)
    def progress(**kw):save(OUT/'progress.json',dict(status='running',pid=os.getpid(),seconds=round(time.monotonic()-started,2),**kw))
    try:
        folder=OUT/'heads_oof';folder.mkdir()
        for fold in range(3):
            progress(phase='fixed400_relative_regression_source_OOF',fold=fold)
            held=regression_folds==fold
            assert {r['image'] for r in training_records if r['fold']==fold}.isdisjoint({r['image'] for r in training_records if r['fold']!=fold})
            head=fit(features[~held],regression_targets[~held]);path=folder/('fold'+str(fold)+'.pt')
            torch.save(head.state_dict(),path);heads[fold]=head;head_sha[fold]=sha(path);pins[str(path)]=sha(path)
            with torch.inference_mode():
                positive_predictions[held]=head(features[held])
                native_predictions[native_folds==fold]=head(native[native_folds==fold])
        positive_before=[r['initial_iou'] for r in training_records]
        positive_after=[overlap(decode(r['box'],p.tolist()),r['target_box']) for r,p in zip(training_records,positive_predictions)]
        positive_summary=dict(examples=len(positive_before),original_mean_iou=sum(positive_before)/len(positive_before),
            refined_mean_iou=sum(positive_after)/len(positive_after),
            newly_below_iou50=sum(a>=.5 and b<.5 for a,b in zip(positive_before,positive_after)),
            not_recognition_accuracy=True)
        indexed={r['image']:r for r in load(PREP/'index.json')['records']}
        referencepath=DATA/'images/train01/normal_073.JPG';assert sha(referencepath)==REFERENCE_SHA
        pins[str(referencepath)]=REFERENCE_SHA;reference=read_image(referencepath)
        cases=[];destination=OUT/'train';destination.mkdir()
        for i,name in enumerate(names):
            progress(phase='all192_original_and_refined_geometry_coverage',completed=i,total=192,image=name)
            currentpath=CURRENT/(Path(name).stem+'_predictions.json');pins[str(currentpath)]=sha(currentpath);case=load(currentpath)
            current=case['trial'];h,w=2736,3648
            source=DATA/'images/train01'/name;pins[str(source)]=sha(source)
            path=Path(indexed[name]['path']);assert sha(path)==indexed[name]['sha256'];pins[str(path)]=sha(path);paired=load(path)
            assert paired['teacher']['source_sha256']==pins[str(source)]
            alternatives=load(BASE/'train'/(Path(name).stem+'_predictions.json'))['alternative']
            pool=[]
            for model in (paired['teacher'],paired['student'],paired['feature'],alternatives):
                assert model['source_sha256']==pins[str(source)] and model['predictions']['source_shape']==[h,w]
                for row in model['predictions']['merged_predictions']:
                    l,t,r,b=row['box_xyxy']
                    if row['confidence']>.05 and 16<=l<r<=w-16 and 16<=t<b<=h-16:pool.append((model['weight_sha256'],row))
            alignmentpath=ROOT/'artifacts/paired_port_semantics_20261003/features_train'/(Path(name).stem+'_source.json')
            pins[str(alignmentpath)]=sha(alignmentpath);alignment=load(alignmentpath)
            assert alignment['source_sha256']==pins[str(source)]
            before=[];after=[];regressed=[]
            remaining=5-(len(current['all_predictions'])-len(current['primary']));assert remaining>=0
            if remaining and alignment['alignment'].get('alignment_quality',{}).get('reliable'):
                _,mask=expected_in_source(reference,alignment['alignment']['source_to_reference_homography'],(h,w))
                def eligible(row):
                    l,t,r,b=row['box_xyxy']
                    if not (16<=l<r<=w-16 and 16<=t<b<=h-16):return False
                    if any(overlap(row['box_xyxy'],old['box_xyxy'])>=.5 for old in current['all_predictions']):return False
                    votes={digest for digest,raw in pool if raw['class_id']==row['class_id'] and overlap(raw['box_xyxy'],row['box_xyxy'])>=.5}
                    row['semantic_model_vote_sha256']=sorted(votes)
                    return len(votes)>=3 and valid_boxes([row['box_xyxy']],mask)==[0]
                for row in case['proposals']:
                    key=(name,row['class_id'],*row['box_xyxy']);j=feature_index[key]
                    original=copy.deepcopy(row)
                    if eligible(original):before.append(original)
                    refined=copy.deepcopy(row);delta=native_predictions[j].tolist()
                    refined.update(box_xyxy=decode(row['box_xyxy'],delta),relative_box_prediction=delta,
                        regression_head_sha256=head_sha[folds[name]],relative_box_refined_from=row['box_xyxy'])
                    if eligible(refined):after.append(refined)
                    regressed.append(refined)
            oldhits=matches(current['all_predictions'],targets[name])[0]
            missing=set(range(len(targets[name])))-oldhits
            covered=lambda candidates:sorted(t for t in missing if any(p['class_id']==targets[name][t]['class_id'] and overlap(p['box_xyxy'],targets[name][t]['box'])>=.5 for p in candidates))
            oldcovered=covered(before);newcovered=covered(before+after)
            row=dict(image=name,missed_targets=len(missing),original_coverage_targets=oldcovered,
                union_coverage_targets=newcovered,new_potential_targets=sorted(set(newcovered)-set(oldcovered)),
                original_eligible_proposals=len(before),refined_eligible_proposals=len(after))
            cases.append(row);save(destination/(Path(name).stem+'_geometry.json'),dict(image=name,current=current,
                original_eligible=before,refined_eligible=after,regressed=regressed,summary=row,
                head_sha256=head_sha[folds[name]],alignment=alignment['alignment']))
        assert all(sha(Path(p))==v for p,v in pins.items()) and native_pose_runtime_fingerprint(REPO)==frozen
        summary=dict(missed_targets=sum(r['missed_targets'] for r in cases),
            original_possible_coverage=sum(len(r['original_coverage_targets']) for r in cases),
            union_possible_coverage=sum(len(r['union_coverage_targets']) for r in cases),
            newly_possible_targets=sum(len(r['new_potential_targets']) for r in cases))
        assert summary['missed_targets']==49
        result=dict(status='geometry_potential_requires_original_pixel_semantics_and_actual_gates' if summary['newly_possible_targets'] else 'rejected_no_new_geometry_potential',
            summary=summary,positive_OOF=positive_summary,cases=cases,pins=pins,runtime=frozen,
            no_full_head=True,no_validation_read=True,no_classifier_accuracy_measured=True,
            union_coverage_not_budget_allocation_or_reference_gate=True,no_deployment=True,
            detector_and_prefix_not_independent_OOF=True,field_accuracy=False,seconds=round(time.monotonic()-started,2))
        save(OUT/'report.json',result);save(OUT/'progress.json',dict(status=result['status'],seconds=result['seconds'],summary=summary))
        print(dict(status=result['status'],summary=summary,positive_OOF=positive_summary),flush=True)
    except BaseException as error:
        save(OUT/'progress.json',dict(status='failed',error=type(error).__name__+': '+str(error)));raise


if __name__=='__main__':main()
