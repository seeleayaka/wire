"""Independent TRAIN labels, twin tensors, grouped curation and score replay."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,REPO,load,save,sha
from current_port_baseline_audit import BASE,read_targets
from audit_port_multiscale_acceptance import metric,matches
from paired_pose_native_training import curate,native_label
from raw_pose_consensus import select
from inspection_agent.paired_native_pose import native_pose_runtime_fingerprint
SOURCE=ROOT/'artifacts/fine_pose_training_20261004'
RAW=ROOT/'artifacts/fine_native_consensus_20261004/train'
OLD=ROOT/'artifacts/paired_pose_native_training_20261004/training'
OUT=ROOT/'artifacts/fine_pose_training_audit_20261004'

def main():
    if OUT.exists():raise FileExistsError('Preserve independent training audit')
    import torch
    torch.set_num_threads(2)
    result=load(SOURCE/'report.json');assert all(sha(Path(p))==v for p,v in result['pins'].items())
    assert native_pose_runtime_fingerprint(REPO)==result['runtime']
    groups=load(ROOT/'artifacts/port_training_multiscale_20261002/protocol.json')
    names=sorted(groups['train_sources']);foldmap={name:i%3 for i,name in enumerate(names)}
    assert set(names).isdisjoint(groups['inner_val_sources'])
    data=torch.load(SOURCE/'training/raw_features.pt',map_location='cpu',weights_only=True)
    raw,twins=data['features'],data['reference_self_features'];assert raw.shape==twins.shape==(969,6144)
    expected=raw[:,1536:3072]
    assert torch.equal(twins,torch.cat((expected,expected,torch.zeros_like(expected),expected*expected),dim=1))
    metadata=load(SOURCE/'training/raw_samples.json');oldmeta=load(OLD/'samples.json')
    selfmeta=[dict(image=r['image'],fold=r['fold'],box=r['box'],label=0,kind='raw_reference_self',observed_is_expected_reference=True) for r in metadata]
    allmeta=oldmeta+metadata+selfmeta;kept,duplicates,conflicts=curate(allmeta,foldmap)
    assert [allmeta[i] for i in kept]==load(SOURCE/'training/samples.json')
    saved=load(SOURCE/'training/curation.json');assert saved['duplicates']==[list(r) for r in duplicates] and saved['conflicts']==conflicts
    training=torch.load(SOURCE/'training/features.pt',map_location='cpu',weights_only=True)
    old=torch.load(OLD/'features.pt',map_location='cpu',weights_only=True)
    assert torch.equal(training['features'],torch.cat((old['features'],raw,twins))[kept])
    assert training['labels'].tolist()==[allmeta[i]['label'] for i in kept] and training['folds'].tolist()==[allmeta[i]['fold'] for i in kept]
    labels=[];pins={};entries=load(BASE/'train/report.json')['cases']
    for entry in entries:
        name=entry['image'];case=load(RAW/(Path(name).stem+'_predictions.json'));targets=read_targets('train',name,[2736,3648],entry['label_sha256'],pins)
        labels.extend(native_label(p,targets) for p in case['proposals'])
    assert labels==[r['label'] for r in metadata]
    rawfolds=torch.tensor([r['fold'] for r in metadata]);oof=torch.zeros((969,3));digests={}
    for fold in range(3):
        head=torch.nn.Linear(6144,3);path=SOURCE/'heads_oof'/('fold'+str(fold)+'.pt');head.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));head.eval()
        digest=sha(path);digests[fold]=digest
        held={r['image'] for r in allmeta if r['fold']==fold};fit={r['image'] for r in allmeta if r['fold']!=fold};assert held.isdisjoint(fit)
        with torch.inference_mode():oof[rawfolds==fold]=head(raw[rawfolds==fold]).softmax(dim=1)
    counts={}
    for stage,probabilities in (('source_train_oof',oof),):
        offset=0;rows=load(SOURCE/stage/'report.json')['cases']
        for row,entry in zip(rows,entries):
            assert row['image']==entry['image'];path=SOURCE/stage/(Path(row['image']).stem+'_predictions.json');case=load(path);n=len(case['proposals'])
            expected_probs=torch.tensor(case['probabilities']).reshape(n,3)
            assert torch.allclose(expected_probs,probabilities[offset:offset+n],atol=1e-7,rtol=1e-6)
            offset+=n
            assert select(case['current'],case['proposals'],case['probabilities'],case['head_sha256'])==case['trial']
            targets=read_targets('train',entry['image'],[2736,3648],entry['label_sha256'],pins)
            for version in ('current','trial'):assert metric(case[version]['all_predictions'],targets)==row[version]
            oldhits=matches(case['current']['all_predictions'],targets)[0];newhits=matches(case['trial']['all_predictions'],targets)[0]
            assert sorted(newhits-oldhits)==row['gained'] and sorted(oldhits-newhits)==row['lost']
        assert offset==969;counts[stage]=len(rows)
    assert all(sha(Path(p))==v for p,v in result['pins'].items()) and native_pose_runtime_fingerprint(REPO)==result['runtime']
    OUT.mkdir();report=dict(status='complete',counts=counts,raw_examples=969,normal_reference_twins=969,
        retained_training_examples=len(kept),duplicates=len(duplicates),conflicts=len(conflicts),
        source_group_twins_excluded_together=True,independent_train_label_tensor_and_score_replay=True,
        field_accuracy=False,no_validation_labels_read=True,no_deployment=True,source_report_sha256=sha(SOURCE/'report.json'))
    save(OUT/'report.json',report);print(report)

if __name__=='__main__':main()
