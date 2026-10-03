"""Exact full fixed classifier TRAIN192, not substituting OOF head scores."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,REPO,OUT,PROPOSALS,load,save,sha
from current_port_baseline_audit import BASE,read_targets
from paired_port_semantic_selection import select
from inspection_agent.resolution_loose_plug_support import resolution_runtime_fingerprint
from audit_port_multiscale_acceptance import metric,matches
NEW=ROOT/'artifacts/paired_semantic_reference_veto_20261003'


def main():
    import torch
    torch.set_num_threads(2)
    trained=load(OUT/'head_full/report.json');prepared=load(OUT/'features_train/report.json')
    weight=Path(trained['checkpoint']['path']);assert sha(weight)==trained['checkpoint']['sha256']
    source=OUT/'features_train';assert sha(source/'features.pt')==prepared['aggregate_feature_sha256']
    assert sha(source/'samples.json')==prepared['samples_sha256']
    assert resolution_runtime_fingerprint(REPO)==prepared['runtime_fingerprint']
    metadata=load(source/'samples.json');features=torch.load(source/'features.pt',map_location='cpu',weights_only=True)['features']
    checkpoint=torch.load(weight,map_location='cpu',weights_only=True);head=torch.nn.Linear(6144,3)
    head.load_state_dict(checkpoint['state_dict'],strict=True);head.requires_grad_(False).eval()
    with torch.inference_mode():probabilities=head(features).softmax(dim=1).tolist()
    destination=NEW/'full_train';destination.mkdir(parents=True,exist_ok=False)
    pins={str(p):sha(p) for p in (Path(__file__),weight,source/'features.pt',source/'samples.json',source/'report.json',
        BASE/'train/report.json',ROOT/'artifacts/paired_semantic_reference_veto_preregistration_20261003/PLAN.md')}
    source_records={r['image']:r for r in prepared['sources']};entries=load(BASE/'train/report.json')['cases'];records=[]
    for entry in entries:
        name=entry['image'];candidate_path=PROPOSALS/('train_'+Path(name).stem+'_proposals.json')
        pins[str(candidate_path)]=sha(candidate_path);current=load(candidate_path)['current']
        indices=[i for i,r in enumerate(metadata) if r['image']==name and r['kind']=='novel_weak_proposal']
        native=[metadata[i]['proposal'] for i in indices];scores=[probabilities[i] for i in indices]
        trial=select(current,native,scores,trained['checkpoint']['sha256'])
        path=destination/(Path(name).stem+'_predictions.json')
        save(path,dict(image=name,current=current,trial=trial,proposals=native,probabilities=scores,
            alignment=source_records[name]['alignment'],source_sha256=source_records[name]['source_sha256'],
            head_sha256=trained['checkpoint']['sha256']))
        targets=read_targets('train',name,[2736,3648],entry['label_sha256'],pins)
        old,new=current['all_predictions'],trial['all_predictions'];oh,nh=matches(old,targets)[0],matches(new,targets)[0]
        records.append(dict(image=name,current=metric(old,targets),trial=metric(new,targets),gained=sorted(nh-oh),lost=sorted(oh-nh),
                            additions=len(trial['paired_semantic_additions']),predictions=str(path)))
    totals={v:{k:sum(r[v][k] for r in records) for k in ('tp','unmatched','fn','predictions','targets')} for v in ('current','trial')}
    assert totals['current']==load(BASE/'train/report.json')['summary']['trial']
    assert {p:sha(Path(p)) for p in pins}==pins and resolution_runtime_fingerprint(REPO)==prepared['runtime_fingerprint']
    save(destination/'report.json',dict(status='complete',summary=totals,cases=records,pins=pins,
        head_sha256=trained['checkpoint']['sha256'],source_features_cached=True,no_reference_veto_yet=True,
        trained_on_all192=True,no_deployment=True,field_accuracy=False))
    print(str(totals))


if __name__=='__main__':main()
