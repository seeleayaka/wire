"""Independent cached fold-head scores and all192 source weak-GT replay."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,REPO,load,save,sha
from current_port_baseline_audit import BASE,read_targets
from inspection_agent.paired_native_pose import native_pose_runtime_fingerprint
from inspection_agent.paired_native_pose_features import select
from audit_port_multiscale_acceptance import metric,matches
from tiny_paired_semantics import TinyPairedHead
SOURCE=ROOT/'artifacts/fine_nonlinear_semantics_20261004'
OLD=ROOT/'artifacts/fine_pose_training_20261004/training'
OUT=ROOT/'artifacts/fine_nonlinear_semantics_replay_20261004'


def main():
    import torch
    torch.set_num_threads(2)
    if OUT.exists():raise FileExistsError('Preserve nonlinear head replay')
    report=load(SOURCE/'report.json');protocol=load(SOURCE/'protocol.json')
    assert report['status']=='rejected' and report['failed_stage']=='source_train_oof'
    assert not (SOURCE/'full').exists()
    assert report['validation_labels_not_read'] and report['classifier_only_source_OOF']
    assert all(sha(Path(p))==v for p,v in report['pins'].items())
    assert native_pose_runtime_fingerprint(REPO)==report['runtime']
    groups=load(ROOT/'artifacts/port_training_multiscale_20261002/protocol.json')
    foldmap={name:i%3 for i,name in enumerate(sorted(groups['train_sources']))}
    assert foldmap==protocol['source_group_folds'] and len(foldmap)==192
    records=load(OLD/'raw_samples.json');samples=load(OLD/'samples.json')
    data=torch.load(OLD/'features.pt',map_location='cpu',weights_only=True)
    raw=torch.load(OLD/'raw_features.pt',map_location='cpu',weights_only=True)['features']
    assert len(samples)==9808 and raw.shape==(969,6144)
    assert all(r['fold']==foldmap[r['image']]==int(data['folds'][i]) and r['label']==int(data['labels'][i]) for i,r in enumerate(samples))
    assert all(r['fold']==foldmap[r['image']] for r in records)
    heads={};scores=torch.zeros((969,3))
    for fold in range(3):
        fit_sources={r['image'] for r in samples if r['fold']!=fold}
        excluded_sources={r['image'] for r in samples if r['fold']==fold}
        assert fit_sources.isdisjoint(excluded_sources)
        assert {r['image'] for r in records if r['fold']==fold}.issubset(excluded_sources)
        path=SOURCE/'heads_oof'/('fold'+str(fold)+'.pt')
        head=TinyPairedHead().eval().requires_grad_(False)
        head.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
        heads[fold]=sha(path)
        indices=torch.tensor([i for i,r in enumerate(records) if r['fold']==fold])
        with torch.inference_mode():scores[indices]=head(raw[indices]).softmax(dim=1)
    saved=load(SOURCE/'source_train_oof/report.json')
    rows=saved['cases'];assert len(rows)==192
    entries={r['image']:r for r in load(BASE/'train/report.json')['cases']}
    pins={};offset=0;added=0
    for row in rows:
        name=row['image'];case=load(SOURCE/'source_train_oof'/(Path(name).stem+'_predictions.json'))
        parent=load(ROOT/'artifacts/fine_native_consensus_20261004/train'/(Path(name).stem+'_predictions.json'))
        assert case['current']==parent['current'] and case['proposals']==parent['proposals']
        count=len(case['proposals']);local=records[offset:offset+count]
        assert len(local)==count and all(r['image']==name and r['box']==p['box_xyxy'] for r,p in zip(local,case['proposals']))
        probabilities=scores[offset:offset+count];offset+=count
        assert torch.allclose(probabilities,torch.tensor(case['probabilities']).reshape(count,3),atol=1e-6,rtol=1e-5)
        digest=heads[foldmap[name]];assert digest==case['head_sha256']
        assert select(case['current'],case['proposals'],case['probabilities'],digest)==case['trial']
        oldrows=case['current']['all_predictions'];newrows=case['trial']['all_predictions']
        assert newrows[:len(oldrows)]==oldrows and len(newrows)<=len(case['trial']['primary'])+5
        added+=len(newrows)-len(oldrows)
        targets=read_targets('train',name,[2736,3648],entries[name]['label_sha256'],pins)
        assert row['current']==metric(oldrows,targets) and row['trial']==metric(newrows,targets)
        old=matches(oldrows,targets)[0];new=matches(newrows,targets)[0]
        assert row['gained']==sorted(new-old) and row['lost']==sorted(old-new)
    assert offset==969
    summary={version:{k:sum(r[version][k] for r in rows) for k in ('tp','unmatched','fn','predictions','targets')} for version in ('current','trial')}
    assert summary==saved['summary']==report['stages']['source_train_oof']['summary']
    assert not saved['qualifies']
    assert all(sha(Path(p))==v for p,v in pins.items())
    assert all(sha(Path(p))==v for p,v in report['pins'].items())
    assert native_pose_runtime_fingerprint(REPO)==report['runtime']
    OUT.mkdir();result=dict(status='pass',experiment_status=report['status'],train_sources=192,
        raw_examples=969,curated_examples=9808,additions=added,summary=summary,
        source_group_tensor_score_and_GT_replay=True,no_full_head_or_validation=True,
        source_report_sha256=sha(SOURCE/'report.json'),auditor_sha256=sha(Path(__file__)),
        no_deployment=True,not_whole_detector_independent_OOF=True,field_accuracy=False)
    save(OUT/'report.json',result);print(result,flush=True)


if __name__=='__main__':main()
