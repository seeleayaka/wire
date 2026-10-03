"""One source-grouped nonlinear head trial; audited cached TRAIN tensors only."""
import os
import sys
import time
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,REPO,load,save,sha
from current_port_baseline_audit import BASE,read_targets
from inspection_agent.paired_native_pose import native_pose_runtime_fingerprint
from inspection_agent.paired_native_pose_features import select
from audit_port_multiscale_acceptance import metric,matches
from tiny_paired_semantics import fit
OLD=ROOT/'artifacts/fine_pose_training_20261004'
SOURCE=ROOT/'artifacts/fine_native_consensus_20261004/train'
OUT=ROOT/'artifacts/fine_nonlinear_semantics_20261004'
PLAN=ROOT/'artifacts/fine_nonlinear_semantics_preregistration_20261004/PLAN.md'


def main():
    import psutil
    import torch
    torch.set_num_threads(2)
    if OUT.exists():raise FileExistsError('Preserve fixed nonlinear source trial')
    assert psutil.virtual_memory().available>6*2**30,'No competing memory-heavy training'
    auditpath=ROOT/'artifacts/fine_pose_training_audit_20261004/report.json'
    audit=load(auditpath);oldreport=load(OLD/'report.json')
    assert audit['status']=='complete' and audit['source_report_sha256']==sha(OLD/'report.json')
    assert oldreport['status']=='rejected' and oldreport['failed_stage']=='source_train_oof'
    assert all(sha(Path(p))==v for p,v in oldreport['pins'].items())
    frozen=native_pose_runtime_fingerprint(REPO)
    assert frozen==oldreport['runtime']
    groupspath=ROOT/'artifacts/port_training_multiscale_20261002/protocol.json'
    names=sorted(load(groupspath)['train_sources']);assert len(names)==192
    source_folds={name:i%3 for i,name in enumerate(names)}
    training=OLD/'training'
    pins={str(p):sha(p) for p in (Path(__file__),Path(__file__).with_name('tiny_paired_semantics.py'),
        PLAN,auditpath,OLD/'report.json',groupspath,training/'features.pt',training/'samples.json',
        training/'raw_features.pt',training/'raw_samples.json')}
    data=torch.load(training/'features.pt',map_location='cpu',weights_only=True)
    raw=torch.load(training/'raw_features.pt',map_location='cpu',weights_only=True)['features']
    samples=load(training/'samples.json');records=load(training/'raw_samples.json')
    features,labels,folds=data['features'],data['labels'],data['folds']
    assert features.shape==(9808,6144) and raw.shape==(969,6144)
    assert len(samples)==9808 and len(records)==969
    assert torch.isfinite(features).all() and torch.isfinite(raw).all()
    assert all(r['fold']==source_folds[r['image']]==int(folds[i]) and r['label']==int(labels[i]) for i,r in enumerate(samples))
    assert all(r['fold']==source_folds[r['image']] for r in records)
    OUT.mkdir();started=time.monotonic();stages={}
    save(OUT/'protocol.json',dict(pins=pins,runtime=frozen,architecture=[6144,16,3],dropout=.1,
        seed=0,steps=400,learning_rate=.01,weight_decay=.001,source_group_folds=source_folds,
        validation_labels_not_read=True,no_training_images_added=True,no_deployment=True,field_accuracy=False))
    def progress(**kw):save(OUT/'progress.json',dict(status='running',pid=os.getpid(),seconds=round(time.monotonic()-started,2),**kw))
    def evaluate(stage,scores,digests):
        folder=OUT/stage;folder.mkdir();rows=[];offset=0
        for entry in load(BASE/'train/report.json')['cases']:
            name=entry['image'];path=SOURCE/(Path(name).stem+'_predictions.json');pins[str(path)]=sha(path);case=load(path)
            n=len(case['proposals']);local=records[offset:offset+n]
            assert all(r['image']==name and r['box']==p['box_xyxy'] for r,p in zip(local,case['proposals'])) and len(local)==n
            values=scores[offset:offset+n].tolist();offset+=n
            digest=digests[source_folds[name]]
            current=case['current'];trial=select(current,case['proposals'],values,digest)
            targets=read_targets('train',name,[2736,3648],entry['label_sha256'],pins)
            old=matches(current['all_predictions'],targets)[0];new=matches(trial['all_predictions'],targets)[0]
            rows.append(dict(image=name,current=metric(current['all_predictions'],targets),trial=metric(trial['all_predictions'],targets),gained=sorted(new-old),lost=sorted(old-new)))
            save(folder/(Path(name).stem+'_predictions.json'),dict(image=name,current=current,trial=trial,
                proposals=case['proposals'],probabilities=values,head_sha256=digest))
        assert offset==len(scores)==969 and len(rows)==192
        totals={version:{k:sum(r[version][k] for r in rows) for k in ('tp','unmatched','fn','predictions','targets')} for version in ('current','trial')}
        assert totals['current']['tp']==295 and totals['current']['unmatched']==4
        normal=sum(r['trial']['predictions'] for r in rows if r['image'].startswith('normal_'))
        qualifies=totals['trial']['tp']>295 and totals['trial']['unmatched']<=4 and normal==0 and not any(r['lost'] for r in rows)
        summary=dict(qualifies=qualifies,summary=totals,normal_cues=normal)
        save(folder/'report.json',dict(status='complete',**summary,cases=rows));stages[stage]=summary
        print(dict(stage=stage,**summary),flush=True)
        return qualifies
    def finish(failed):
        assert all(sha(Path(p))==v for p,v in pins.items()) and native_pose_runtime_fingerprint(REPO)==frozen
        result=dict(status='rejected' if failed else 'TRAIN_pass_requires_fresh_holdouts_and_actual_gates',
            failed_stage=failed,stages=stages,pins=pins,runtime=frozen,no_deployment=True,
            validation_labels_not_read=True,classifier_only_source_OOF=True,field_accuracy=False,
            seconds=round(time.monotonic()-started,2))
        save(OUT/'report.json',result);save(OUT/'progress.json',dict(status=result['status'],seconds=result['seconds']))
        print(dict(status=result['status'],seconds=result['seconds']),flush=True)
    try:
        rawfolds=torch.tensor([r['fold'] for r in records]);oof=torch.zeros((969,3));digests={}
        heads=OUT/'heads_oof';heads.mkdir()
        for fold in range(3):
            held=folds==fold
            assert {r['image'] for r in samples if r['fold']==fold}.isdisjoint({r['image'] for r in samples if r['fold']!=fold})
            progress(phase='fixed400_nonlinear_source_OOF',fold=fold)
            head=fit(features[~held],labels[~held]);path=heads/('fold'+str(fold)+'.pt')
            torch.save(head.state_dict(),path);digests[fold]=sha(path);pins[str(path)]=digests[fold]
            with torch.inference_mode():oof[rawfolds==fold]=head(raw[rawfolds==fold]).softmax(dim=1)
        if not evaluate('source_train_oof',oof,digests):finish('source_train_oof');return
        progress(phase='fixed400_nonlinear_full_head');head=fit(features,labels)
        folder=OUT/'full';folder.mkdir();path=folder/'last_head.pt'
        torch.save(dict(state_dict=head.state_dict(),architecture=[6144,16,3],dropout=.1,
            input_dimensions=6144,classes=['other','unplugged_plug','unplugged_jack'],encoder_sha256=frozen['encoder']),path)
        digest=sha(path);pins[str(path)]=digest
        with torch.inference_mode():scores=head(raw).softmax(dim=1)
        if not evaluate('full_train',scores,{i:digest for i in range(3)}):finish('full_train');return
        finish(None)
    except BaseException as error:
        save(OUT/'progress.json',dict(status='failed',error=type(error).__name__+': '+str(error)));raise


if __name__=='__main__':main()
