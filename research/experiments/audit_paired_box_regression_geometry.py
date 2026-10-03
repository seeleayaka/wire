"""Independent tensor/fold/relative-geometry/vote/weak-coverage replay."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,REPO,DATA,load,save,sha,read_image
from current_port_baseline_audit import BASE,read_targets
from inspection_agent.paired_native_pose import native_pose_runtime_fingerprint
from inspection_agent.paired_port_features import expected_in_source,valid_boxes
from audit_port_multiscale_acceptance import overlap,matches
from relative_port_box import encode,decode,bounded
SOURCE=ROOT/'artifacts/paired_box_regression_geometry_20261004'
OUT=ROOT/'artifacts/paired_box_regression_geometry_replay_20261004'


def main():
    import cv2
    import torch
    cv2.setNumThreads(1);torch.set_num_threads(2)
    if OUT.exists():raise FileExistsError('Preserve independent relative geometry replay')
    report=load(SOURCE/'report.json');protocol=load(SOURCE/'protocol.json')
    assert report['status'] in ('geometry_potential_requires_original_pixel_semantics_and_actual_gates','rejected_no_new_geometry_potential')
    assert report['no_full_head'] and report['no_validation_read'] and report['no_classifier_accuracy_measured']
    assert all(sha(Path(p))==v for p,v in report['pins'].items())
    frozen=native_pose_runtime_fingerprint(REPO);assert report['runtime']==frozen
    supplemental=load(ROOT/'artifacts/paired_box_regression_input_readiness_20261004/report.json')
    assert supplemental['status']=='pass' and supplemental['before_child'] and supplemental['runtime']==frozen
    assert all(sha(Path(p))==v for p,v in supplemental['pins'].items())
    training=ROOT/'artifacts/fine_pose_training_20261004/training'
    data=torch.load(training/'features.pt',map_location='cpu',weights_only=True)
    samples=load(training/'samples.json')
    records=load(SOURCE/'regression_training_records.json')
    nativepath=ROOT/'artifacts/paired_pose_jitter_agreement_20261003'
    native=torch.load(nativepath/'native_features.pt',map_location='cpu',weights_only=True)['features']
    metadata=load(nativepath/'native_samples.json');assert native.shape==(679,6144)
    feature_index={(r['image'],r['proposal']['class_id'],*r['proposal']['box_xyxy']):i for i,r in enumerate(metadata)}
    names=sorted(protocol['source_folds']);foldmap={name:i%3 for i,name in enumerate(names)}
    assert foldmap==protocol['source_folds'] and len(names)==192
    assert all(r['fold']==foldmap[r['image']] for r in metadata)
    assert [r['sample_index'] for r in records]==[i for i,r in enumerate(samples) if r['label']>0]
    entries={r['image']:r for r in load(BASE/'train/report.json')['cases']};pins={}
    targets={name:read_targets('train',name,[2736,3648],entries[name]['label_sha256'],pins) for name in names}
    for row in records:
        sample=samples[row['sample_index']]
        assert sample['image']==row['image'] and sample['box']==row['box'] and sample['label']==row['label']
        assert sample['fold']==row['fold']==foldmap[row['image']]==int(data['folds'][row['sample_index']])
        closest=max((overlap(row['box'],t['box']),j,t) for j,t in enumerate(targets[row['image']]) if t['class_id']==row['label']-1)
        assert closest[0]==row['initial_iou'] and closest[1]==row['target_index'] and closest[2]['box']==row['target_box']
        assert bounded(encode(row['box'],row['target_box']))==row['relative_target']
    positive_features=data['features'][[r['sample_index'] for r in records]]
    positive_predictions=torch.zeros((len(records),4));native_predictions=torch.zeros((679,4));digests={}
    for fold in range(3):
        held=[i for i,r in enumerate(records) if r['fold']==fold]
        train_sources={r['image'] for r in records if r['fold']!=fold}
        assert train_sources.isdisjoint({r['image'] for r in records if r['fold']==fold})
        path=SOURCE/'heads_oof'/('fold'+str(fold)+'.pt');head=torch.nn.Linear(6144,4).eval().requires_grad_(False)
        head.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));digests[fold]=sha(path)
        native_held=[i for i,r in enumerate(metadata) if r['fold']==fold]
        with torch.inference_mode():
            positive_predictions[held]=head(positive_features[held])
            native_predictions[native_held]=head(native[native_held])
    before=[r['initial_iou'] for r in records]
    after=[overlap(decode(r['box'],p.tolist()),r['target_box']) for r,p in zip(records,positive_predictions)]
    assert report['positive_OOF']==dict(examples=len(before),original_mean_iou=sum(before)/len(before),
        refined_mean_iou=sum(after)/len(after),newly_below_iou50=sum(a>=.5 and b<.5 for a,b in zip(before,after)),not_recognition_accuracy=True)
    reference=read_image(DATA/'images/train01/normal_073.JPG')
    cases=[]
    for name in names:
        saved=load(SOURCE/'train'/(Path(name).stem+'_geometry.json'))
        currentcase=load(ROOT/'artifacts/paired_pose_native_three_20261004/full_train'/(Path(name).stem+'_predictions.json'))
        assert currentcase['trial']==saved['current']
        paired=load(ROOT/'artifacts/paired_support_graph_20261003/source_selections/train'/(Path(name).stem+'.json'))
        altpath=BASE/'train'/(Path(name).stem+'_predictions.json')
        assert str(altpath) in supplemental['pins'] and sha(altpath)==supplemental['pins'][str(altpath)]
        alternative=load(altpath)['alternative']
        pool=[(model['weight_sha256'],row) for model in (paired['teacher'],paired['student'],paired['feature'],alternative)
            for row in model['predictions']['merged_predictions'] if row['confidence']>.05
            and 16<=row['box_xyxy'][0]<row['box_xyxy'][2]<=3648-16 and 16<=row['box_xyxy'][1]<row['box_xyxy'][3]<=2736-16]
        before=[];after=[];regressed=[]
        remaining=5-(len(saved['current']['all_predictions'])-len(saved['current']['primary']))
        if remaining and saved['alignment'].get('alignment_quality',{}).get('reliable'):
            _,mask=expected_in_source(reference,saved['alignment']['source_to_reference_homography'],(2736,3648))
            def eligible(row):
                l,t,r,b=row['box_xyxy']
                if not (16<=l<r<=3648-16 and 16<=t<b<=2736-16):return False
                if any(overlap(row['box_xyxy'],old['box_xyxy'])>=.5 for old in saved['current']['all_predictions']):return False
                votes={weight for weight,raw in pool if raw['class_id']==row['class_id'] and overlap(raw['box_xyxy'],row['box_xyxy'])>=.5}
                return len(votes)>=3 and valid_boxes([row['box_xyxy']],mask)==[0]
            import copy
            for original,new in zip(currentcase['proposals'],saved['regressed']):
                j=feature_index[(name,original['class_id'],*original['box_xyxy'])]
                prediction=native_predictions[j].tolist()
                assert prediction==new['relative_box_prediction'] and decode(original['box_xyxy'],prediction)==new['box_xyxy']
                assert new['relative_box_refined_from']==original['box_xyxy'] and new['regression_head_sha256']==digests[foldmap[name]]
                if eligible(original):before.append(original)
                if eligible(new):
                    votes=sorted({weight for weight,raw in pool if raw['class_id']==new['class_id'] and overlap(raw['box_xyxy'],new['box_xyxy'])>=.5})
                    assert votes==new['semantic_model_vote_sha256'];after.append(new)
            assert len(currentcase['proposals'])==len(saved['regressed'])
        else:assert not saved['regressed']
        assert [(p['class_id'],p['box_xyxy']) for p in before]==[(p['class_id'],p['box_xyxy']) for p in saved['original_eligible']]
        assert after==saved['refined_eligible']
        oldhits=matches(saved['current']['all_predictions'],targets[name])[0];missing=set(range(len(targets[name])))-oldhits
        covered=lambda candidates:sorted(t for t in missing if any(p['class_id']==targets[name][t]['class_id'] and overlap(p['box_xyxy'],targets[name][t]['box'])>=.5 for p in candidates))
        oldcovered=covered(before);newcovered=covered(before+after)
        row=dict(image=name,missed_targets=len(missing),original_coverage_targets=oldcovered,union_coverage_targets=newcovered,
            new_potential_targets=sorted(set(newcovered)-set(oldcovered)),original_eligible_proposals=len(before),refined_eligible_proposals=len(after))
        assert row==saved['summary'];cases.append(row)
    assert cases==report['cases']
    summary=dict(missed_targets=sum(r['missed_targets'] for r in cases),original_possible_coverage=sum(len(r['original_coverage_targets']) for r in cases),
        union_possible_coverage=sum(len(r['union_coverage_targets']) for r in cases),newly_possible_targets=sum(len(r['new_potential_targets']) for r in cases))
    assert summary==report['summary']
    assert all(sha(Path(p))==v for p,v in report['pins'].items()) and all(sha(Path(p))==v for p,v in supplemental['pins'].items())
    assert native_pose_runtime_fingerprint(REPO)==frozen
    OUT.mkdir();result=dict(status='pass',train_sources=192,cached_native_features=679,
        positive_examples=len(records),summary=summary,independent_regression_tensor_head_vote_geometry_and_GT_replay=True,
        supplemental_all192_alternative_input_hashes_verified=True,source_report_sha256=sha(SOURCE/'report.json'),
        no_classifier_accuracy_measured=True,no_deployment=True,field_accuracy=False)
    save(OUT/'report.json',result);print(result,flush=True)


if __name__=='__main__':main()
