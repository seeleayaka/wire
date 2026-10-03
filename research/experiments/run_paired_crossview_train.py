"""OOF crop, OOF192 then exact full pair192, no descriptor re-extraction."""
import os,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,REPO,OUT as CONTEXT,load,save,sha
from prepare_paired_semantic_geometry import NEW as GEOMETRY
from current_port_baseline_audit import BASE,read_targets
from audit_port_multiscale_acceptance import metric,matches
from port_semantic_verifier import fit_head
from paired_crossview_selection import select
from inspection_agent.paired_port_geometry import paired_runtime_fingerprint
FOOTPRINT=ROOT/'artifacts/paired_boxpool_original_rows_20261003'
OUT=ROOT/'artifacts/paired_crossview_20261003'


def identity(row):return row['image'],row['kind'],tuple(row['box']),row['label'],row['fold']


def main():
    if OUT.exists():raise FileExistsError('Preserve fixed conjunction experiment')
    import torch
    torch.set_num_threads(2);started=time.monotonic();frozen=paired_runtime_fingerprint(REPO)
    metadata=load(FOOTPRINT/'features_train/samples.json');oldmeta=load(GEOMETRY/'features_train/samples.json')
    prepared=load(FOOTPRINT/'features_train/report.json');oldprep=load(GEOMETRY/'features_train/report.json')
    assert len(metadata)==len(oldmeta)==3134 and prepared['accepted_paired_runtime']==frozen
    for root,report in ((FOOTPRINT,prepared),(GEOMETRY,oldprep)):
        assert sha(root/'features_train/features.pt')==report['aggregate_feature_sha256']
        assert sha(root/'features_train/samples.json')==report['samples_sha256']
    index={identity(r):i for i,r in enumerate(oldmeta)};assert all(identity(r) in index for r in metadata)
    olddata=torch.load(GEOMETRY/'features_train/features.pt',map_location='cpu',weights_only=True)
    data=torch.load(FOOTPRINT/'features_train/features.pt',map_location='cpu',weights_only=True)
    context=olddata['features'][[index[identity(r)] for r in metadata]];footprint=data['features'];labels=data['labels'];folds=data['folds']
    assert torch.equal(labels,olddata['labels'][[index[identity(r)] for r in metadata]])
    foot_report=load(FOOTPRINT/'heads_oof/report.json');foot_probs_path=FOOTPRINT/'heads_oof/oof_probabilities.pt'
    assert sha(foot_probs_path)==foot_report['probabilities_sha256']
    footprint_probs=torch.load(foot_probs_path,map_location='cpu',weights_only=True);context_probs=torch.zeros_like(footprint_probs);head_shas=[]
    pins={str(p):sha(p) for p in (Path(__file__),Path(__file__).with_name('paired_crossview_selection.py'),Path(__file__).with_name('port_semantic_verifier.py'),
        ROOT/'artifacts/paired_crossview_preregistration_20261003/PLAN.md',foot_probs_path,FOOTPRINT/'heads_oof/report.json',
        FOOTPRINT/'features_train/features.pt',FOOTPRINT/'features_train/samples.json',GEOMETRY/'features_train/features.pt',GEOMETRY/'features_train/samples.json')}
    OUT.mkdir();save(OUT/'protocol.json',dict(pins=pins,runtime_fingerprint=frozen,member_context1107_footprint3134=True,
        paired_classifier_OOF_only=True,min_member_not_calibrated=True,no_score_or_GT_change=True,no_deployment=True,field_accuracy=False))
    for fold in range(3):
        path=CONTEXT/'heads_oof'/f'fold{fold}_head.pt';fp=FOOTPRINT/'heads_oof'/f'fold{fold}_head.pt'
        pins[str(path)]=sha(path);pins[str(fp)]=sha(fp);head=torch.nn.Linear(6144,3)
        head.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));head.eval().requires_grad_(False)
        with torch.inference_mode():context_probs[folds==fold]=head(context[folds==fold]).softmax(dim=1)
        head_shas.append([sha(path),sha(fp)])
    cs,cc=context_probs.max(1);fs,fc=footprint_probs.max(1);accepted=(cc==fc)&(cc>0)&(cs>=.98)&(fs>=.98);correct=accepted&(cc==labels)
    gt=torch.tensor([r['kind']=='gt_port' for r in metadata]);tp=int(correct.sum());fp=int((accepted&~correct).sum());gt_correct=int((correct&gt).sum())
    crop=dict(tp=tp,unmatched=fp,precision=tp/max(1,tp+fp),GT_correct=gt_correct,GT_targets=344,recall=gt_correct/344)
    crop['qualifies']=crop['precision']>=.98 and crop['recall']>=.25
    save(OUT/'crop_report.json',dict(status='complete',summary=crop,source_classifier_OOF_only=True))
    save(OUT/'progress.json',dict(status='running',pid=os.getpid(),phase='crop_gate',summary=crop));print(str(crop),flush=True)
    if not crop['qualifies']:
        save(OUT/'progress.json',dict(status='rejected',phase='crop_gate',summary=crop));return
    def source_stage(stage,context_values,footprint_values,digests):
        folder=OUT/stage;folder.mkdir();records=[]
        for entry in load(BASE/'train/report.json')['cases']:
            name=entry['image'];path=GEOMETRY/'full_train'/(Path(name).stem+'_predictions.json');pins[str(path)]=sha(path);current=load(path)['trial']
            indices=[i for i,r in enumerate(metadata) if r['image']==name and r['kind']=='novel_weak_proposal']
            native=[metadata[i]['proposal'] for i in indices];fold=metadata[indices[0]]['fold'] if indices else 0
            heads=digests[fold] if stage=='source_train_oof' else digests
            probabilities=[context_values[indices].tolist(),footprint_values[indices].tolist()]
            trial=select(current,native,probabilities,heads)
            save(folder/(Path(name).stem+'_predictions.json'),dict(image=name,current=current,trial=trial,proposals=native,member_probabilities=probabilities))
            targets=read_targets('train',name,[2736,3648],entry['label_sha256'],pins)
            oh,nh=matches(current['all_predictions'],targets)[0],matches(trial['all_predictions'],targets)[0]
            records.append(dict(image=name,current=metric(current['all_predictions'],targets),trial=metric(trial['all_predictions'],targets),
                gained=sorted(nh-oh),lost=sorted(oh-nh),additions=len(trial['paired_crossview_additions'])))
        totals={kind:{k:sum(r[kind][k] for r in records) for k in ('tp','unmatched','fn','predictions','targets')} for kind in ('current','trial')}
        assert totals['current']==load(GEOMETRY/'full_train/report.json')['summary']['trial']
        normal=sum(r['trial']['predictions'] for r in records if r['image'].startswith('normal_'))
        qualifies=totals['trial']['tp']>totals['current']['tp'] and totals['trial']['unmatched']<=totals['current']['unmatched'] and not any(r['lost'] for r in records) and normal==0
        save(folder/'report.json',dict(status='complete',qualifies=qualifies,summary=totals,cases=records,normal_cues=normal,field_accuracy=False))
        print(str(dict(stage=stage,qualifies=qualifies,summary=totals)),flush=True);return qualifies
    save(OUT/'progress.json',dict(status='running',pid=os.getpid(),phase='source_OOF'))
    if not source_stage('source_train_oof',context_probs,footprint_probs,head_shas):
        save(OUT/'progress.json',dict(status='rejected',phase='source_OOF'));return
    foot_head=fit_head(footprint,labels);head_path=OUT/'footprint_full_head.pt';torch.save(dict(state_dict=foot_head.state_dict(),
        input_dimensions=6144,encoder_sha256=prepared['encoder_sha256'],fixed_steps=400),head_path)
    context_path=CONTEXT/'head_full/last_head.pt';pins[str(context_path)]=sha(context_path);head=torch.nn.Linear(6144,3)
    checkpoint=torch.load(context_path,map_location='cpu',weights_only=True);head.load_state_dict(checkpoint['state_dict']);head.eval().requires_grad_(False)
    with torch.inference_mode():context_full=head(context).softmax(1);footprint_full=foot_head(footprint).softmax(1)
    full_shas=[sha(context_path),sha(head_path)]
    qualifies=source_stage('full_train',context_full,footprint_full,full_shas)
    assert {p:sha(Path(p)) for p in pins}==pins and paired_runtime_fingerprint(REPO)==frozen
    save(OUT/'full_pair.json',dict(status='complete',qualifies_train=qualifies,context_head=dict(path=str(context_path),sha256=full_shas[0]),
        footprint_head=dict(path=str(head_path),sha256=full_shas[1]),runtime_fingerprint=frozen,pins=pins,
        no_validation_training=True,seconds=round(time.monotonic()-started,2),no_deployment=True,field_accuracy=False))
    save(OUT/'progress.json',dict(status='full_train_pass_requires_fresh_holdouts' if qualifies else 'rejected',phase='full_train',no_deployment=True))

if __name__=='__main__':main()
