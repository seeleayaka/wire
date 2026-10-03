"""Actual frozen89 novel proposal boxes with OOF classifier, ALL192 source gate."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,REPO,OUT,PROPOSALS,load,save,sha
from current_port_baseline_audit import BASE,read_targets
from inspection_agent.resolution_loose_plug_support import resolution_runtime_fingerprint
from audit_port_multiscale_acceptance import metric,matches
from paired_port_semantic_selection import select


def main():
    destination=OUT/'source_train_oof'
    if destination.exists():raise FileExistsError('Preserve actual source gate')
    trained=load(OUT/'heads_oof/report.json');assert trained['status']=='complete' and trained['qualifies_crop_feasibility']
    source=OUT/'features_train';prepared=load(source/'report.json');assert resolution_runtime_fingerprint(REPO)==prepared['runtime_fingerprint']
    import torch
    metadata=load(source/'samples.json');probabilitypath=OUT/'heads_oof/oof_probabilities.pt'
    assert sha(source/'samples.json')==prepared['samples_sha256'] and sha(probabilitypath)==trained['probabilities_sha256']
    probabilities=torch.load(probabilitypath,map_location='cpu',weights_only=True).tolist();assert len(probabilities)==len(metadata)
    source_records={r['image']:r for r in prepared['sources']};pins=dict(trained['pins'],**trained['head_pins'])
    for p in (Path(__file__),Path(__file__).with_name('paired_port_semantic_selection.py'),OUT/'heads_oof/report.json',source/'report.json',probabilitypath):pins[str(p)]=sha(p)
    destination.mkdir();save(destination/'protocol.json',dict(pins=pins,all192_training=True,classifier_OOF_only=True,
        same_native_proposals_before_GT_scoring=True,old_current_prefix_shared5plus5=True,reference_alignment_or_context_failures_explicit_abstentions=True,
        no_model_deployment=True,field_accuracy=False))
    records=[];entries=load(BASE/'train/report.json')['cases'];assert len(entries)==192
    for entry in entries:
        name=entry['image'];path=PROPOSALS/('train_'+Path(name).stem+'_proposals.json');pins[str(path)]=sha(path);candidate=load(path);current=candidate['current']
        indices=[i for i,r in enumerate(metadata) if r['image']==name and r['kind']=='novel_weak_proposal']
        native=[metadata[i]['proposal'] for i in indices];scores=[probabilities[i] for i in indices]
        folded=source_records[name]['fold'];headpath=OUT/'heads_oof'/f'fold{folded}_head.pt';digest=sha(headpath)
        assert digest==trained['head_pins'][str(headpath)]
        trial=select(current,native,scores,digest)
        targets=read_targets('train',name,[2736,3648],entry['label_sha256'],pins)
        old,new=current['all_predictions'],trial['all_predictions'];oh,nh=matches(old,targets)[0],matches(new,targets)[0]
        row=dict(image=name,current=metric(old,targets),trial=metric(new,targets),gained=sorted(nh-oh),lost=sorted(oh-nh),
            source_feature_status=source_records[name]['status'],all_proposals=len(candidate['candidates']),valid_proposals=len(native),additions=len(trial['paired_semantic_additions']))
        records.append(row);save(destination/(Path(name).stem+'_predictions.json'),dict(image=name,current=current,trial=trial,proposals=native,probabilities=scores,record=row))
    totals={v:{k:sum(r[v][k] for r in records) for k in ('tp','unmatched','fn','predictions','targets')} for v in ('current','trial')}
    assert totals['current']==load(BASE/'train/report.json')['summary']['trial']
    normal=sum(r['trial']['predictions'] for r in records if r['image'].startswith('normal_'))
    qualifies=totals['trial']['tp']>totals['current']['tp'] and totals['trial']['unmatched']<=totals['current']['unmatched'] and not any(r['lost'] for r in records) and normal==0
    assert {p:sha(Path(p)) for p in pins}==pins and resolution_runtime_fingerprint(REPO)==prepared['runtime_fingerprint']
    result=dict(status='complete',qualifies=qualifies,summary=totals,cases=records,normal_cues=normal,
        old_current_cached=True,new_proposal_pair_embeddings_newly_extracted=True,classifier_OOF_only=True,
        no_inner_outer_source_accuracy_yet=True,no_model_deployment=True,field_accuracy=False)
    save(destination/'report.json',result);save(destination/'progress.json',dict(status='complete',qualifies=qualifies,summary=totals));print(str(dict(qualifies=qualifies,summary=totals)),flush=True)


if __name__=='__main__':main()
