"""Independent fixed three-checkpoint selection and model identity replay."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,REPO,load,save,sha
from current_port_baseline_audit import BASE,read_targets
from paired_pose_three_vote import select
from audit_port_multiscale_acceptance import metric,matches
from inspection_agent.paired_median_geometry import median_runtime_fingerprint
SOURCE=ROOT/'artifacts/paired_pose_native_three_20261004'
TRAINING=ROOT/'artifacts/paired_pose_native_training_20261004'
OUT=ROOT/'artifacts/paired_pose_native_three_audit_20261004'


def main():
    if OUT.exists():raise FileExistsError('Preserve independent source replay')
    final=load(SOURCE/'report.json');assert final['status'] in ('rejected','source_pass_requires_reference_ROI_SAM_Qt')
    pins=dict(final['pins'])
    for path in (SOURCE/'report.json',Path(__file__),Path(__file__).with_name('paired_pose_three_vote.py')):pins[str(path)]=sha(path)
    assert {p:sha(Path(p)) for p in pins}==pins and median_runtime_fingerprint(REPO)==final['median_runtime_fingerprint']
    names=sorted(load(TRAINING/'protocol.json')['train_sources']);folds={name:i%3 for i,name in enumerate(names)}
    trained=load(TRAINING/'training/samples.json');assert all(r['image'] in folds and r['fold']==folds[r['image']] for r in trained)
    heads={fold:sha(TRAINING/'heads_oof'/f'fold{fold}_head.pt') for fold in range(3)}
    full=sha(TRAINING/'full/last_head.pt');outputs={}
    for stage,record in final['stages'].items():
        dataset='train' if stage in ('source_train_oof','full_train') else stage;rows=[];folder=SOURCE/stage
        entries=load(BASE/dataset/'report.json')['cases'];stage_report=load(folder/'report.json');assert len(entries)==len(stage_report['cases'])
        for entry in entries:
            name=entry['image'];path=folder/(Path(name).stem+'_predictions.json');pins[str(path)]=sha(path);case=load(path)
            # Empty candidate cases have no scored classifier use.
            if case['proposals']:assert case['head_sha256']==(heads[folds[name]] if stage=='source_train_oof' else full)
            repeated=select(case['current'],case['proposals'],case['probabilities'],case['head_sha256']);assert repeated==case['trial'],name
            current=case['current'];assert repeated['all_predictions'][:len(current['all_predictions'])]==current['all_predictions']
            assert len(repeated['primary'])<=5 and len(repeated['all_predictions'])-len(repeated['primary'])<=5
            additions=repeated['paired_semantic_additions'];assert len({p['pose_parent_seed_id'] for p in additions})==len(additions)
            assert all(len(set(p['semantic_model_vote_sha256']))>=3 and p['confidence']>=.98 for p in additions)
            if 'cached_feature_provenance' in case:
                provenance=Path(case['cached_feature_provenance']);assert sha(provenance)==pins[str(provenance)]
                cached=load(provenance);assert case['probabilities']==cached['auxiliary_probabilities'] and case['proposals']==cached['proposals']
            targets=read_targets(dataset,name,[2736,3648],entry['label_sha256'],pins)
            old,new=matches(current['all_predictions'],targets)[0],matches(repeated['all_predictions'],targets)[0]
            rows.append(dict(image=name,current=metric(current['all_predictions'],targets),trial=metric(repeated['all_predictions'],targets),gained=sorted(new-old),lost=sorted(old-new)))
        summary={v:{k:sum(r[v][k] for r in rows) for k in ('tp','unmatched','fn','predictions','targets')} for v in ('current','trial')}
        assert summary==record['summary']==stage_report['summary']
        normal=sum(r['trial']['predictions'] for r in rows if r['image'].startswith('normal_'))
        improved=summary['trial']['tp']>summary['current']['tp'] if stage!='outer' else summary['trial']['tp']>=summary['current']['tp']
        qualifies=improved and summary['trial']['unmatched']<=summary['current']['unmatched'] and not any(r['lost'] for r in rows) and normal==0
        assert qualifies==stage_report['qualifies']==record['qualifies']
        outputs[stage]=dict(qualifies=qualifies,summary=summary,cases=len(rows),normal_cues=normal)
    assert {p:sha(Path(p)) for p in pins}==pins and median_runtime_fingerprint(REPO)==final['median_runtime_fingerprint']
    OUT.mkdir();save(OUT/'report.json',dict(status='complete',stages=outputs,pins=pins,
        all_dictionary_selection_replays_equal=True,source_fold_identity_verified=True,original_native_prefix_preserved=True,
        exact_classified_crop_scores_preserved=True,no_deployment=True,field_accuracy=False))
    print(str(outputs),flush=True)


if __name__=='__main__':main()
