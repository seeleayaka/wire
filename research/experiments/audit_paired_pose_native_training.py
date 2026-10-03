"""Independent replay of complete native-classifier selection and scoring."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,REPO,load,save,sha
from current_port_baseline_audit import BASE,read_targets
from paired_pose_head_agreement import select
from audit_port_multiscale_acceptance import metric,matches
from inspection_agent.paired_port_geometry import HEAD_SHA
from inspection_agent.paired_median_geometry import median_runtime_fingerprint
SOURCE=ROOT/'artifacts/paired_pose_native_training_20261004'
OUT=ROOT/'artifacts/paired_pose_native_training_audit_20261004'


def main():
    if OUT.exists():raise FileExistsError('Preserve independent replay')
    final=load(SOURCE/'report.json');assert final['status'] in ('rejected','source_pass_requires_reference_ROI_SAM_Qt')
    pins=dict(final['pins']);pins[str(SOURCE/'report.json')]=sha(SOURCE/'report.json');pins[str(Path(__file__))]=sha(Path(__file__))
    assert {p:sha(Path(p)) for p in pins}==pins
    assert final['median_runtime_fingerprint']==median_runtime_fingerprint(REPO)
    source_names=sorted(load(SOURCE/'protocol.json')['train_sources']);folds={name:i%3 for i,name in enumerate(source_names)}
    metadata=load(SOURCE/'training/samples.json')
    assert all(r['image'] in folds and r['fold']==folds[r['image']] for r in metadata)
    assert all({r['image'] for r in metadata if r['fold']==fold}.isdisjoint(
        {r['image'] for r in metadata if r['fold']!=fold}) for fold in range(3))
    outputs={}
    for stage,record in final['stages'].items():
        dataset='train' if stage in ('source_train_oof','full_train') else stage
        entries=load(BASE/dataset/'report.json')['cases'];folder=SOURCE/stage;report=load(folder/'report.json')
        assert len(entries)==len(report['cases'])
        rows=[]
        for entry in entries:
            name=entry['image'];path=folder/(Path(name).stem+'_predictions.json');pins[str(path)]=sha(path);case=load(path)
            proposed=select(case['current'],case['proposals'],case['original_probabilities'],case['auxiliary_probabilities'],HEAD_SHA,case['auxiliary_head_sha256'])
            assert proposed==case['trial'],name
            current=case['current'];assert proposed['all_predictions'][:len(current['all_predictions'])]==current['all_predictions']
            assert len(proposed['primary'])<=5 and len(proposed['all_predictions'])-len(proposed['primary'])<=5
            for new in proposed['paired_semantic_additions']:
                assert len(set(new['semantic_model_vote_sha256']))>=2
                assert new['confidence']>=.98 and new['pose_auxiliary_probability']>=.98
            targets=read_targets(dataset,name,[2736,3648],entry['label_sha256'],pins)
            old,new=matches(current['all_predictions'],targets)[0],matches(proposed['all_predictions'],targets)[0]
            rows.append(dict(image=name,current=metric(current['all_predictions'],targets),trial=metric(proposed['all_predictions'],targets),gained=sorted(new-old),lost=sorted(old-new)))
        totals={v:{k:sum(r[v][k] for r in rows) for k in ('tp','unmatched','fn','predictions','targets')} for v in ('current','trial')}
        assert totals==record['summary']==report['summary']
        assert not any(r['lost'] for r in rows)
        outputs[stage]=dict(cases=len(rows),summary=totals,qualifies=report['qualifies'])
    assert {p:sha(Path(p)) for p in pins}==pins and median_runtime_fingerprint(REPO)==final['median_runtime_fingerprint']
    OUT.mkdir();save(OUT/'report.json',dict(status='complete',stages=outputs,pins=pins,source_fold_partition_verified=True,
        all_prediction_dictionary_replays_equal=True,original_native_prefix_preserved=True,no_deployment=True,field_accuracy=False))
    print(str(outputs),flush=True)


if __name__=='__main__':main()
