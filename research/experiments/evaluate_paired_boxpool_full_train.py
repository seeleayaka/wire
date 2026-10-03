"""Score exact final head on all TRAIN sources after accepted geometry prefix."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_boxpool_features import OUT
from prepare_paired_semantic_geometry import NEW as GEOMETRY
from prepare_paired_port_semantics import ROOT,REPO,load,save,sha
from current_port_baseline_audit import BASE,read_targets
from audit_port_multiscale_acceptance import metric,matches
from paired_port_semantic_selection import select
from inspection_agent.paired_port_geometry import paired_runtime_fingerprint


def main():
    destination=OUT/'full_train'
    if destination.exists():raise FileExistsError('Preserve exact final head checks')
    trained=load(OUT/'head_full/report.json');prepared=load(OUT/'features_train/report.json')
    weight=Path(trained['checkpoint']['path']);assert sha(weight)==trained['checkpoint']['sha256'] and trained['no_validation_training']
    frozen=paired_runtime_fingerprint(REPO);assert prepared['accepted_paired_runtime']==frozen
    metadata=load(OUT/'features_train/samples.json');assert sha(OUT/'features_train/features.pt')==prepared['aggregate_feature_sha256']
    pins={str(p):sha(p) for p in (Path(__file__),Path(__file__).with_name('paired_port_semantic_selection.py'),
        OUT/'head_full/report.json',weight,OUT/'features_train/samples.json',OUT/'features_train/features.pt',GEOMETRY/'full_train/report.json',BASE/'train/report.json')}
    destination.mkdir();save(destination/'protocol.json',dict(pins=pins,runtime_fingerprint=frozen,GT_after_selection=True,
        preserve_accepted_geometry_prefix=True,all192=True,no_deployment=True,field_accuracy=False))
    import torch
    torch.set_num_threads(2);checkpoint=torch.load(weight,map_location='cpu',weights_only=True)
    head=torch.nn.Linear(6144,3);head.load_state_dict(checkpoint['state_dict']);head.eval().requires_grad_(False)
    data=torch.load(OUT/'features_train/features.pt',map_location='cpu',weights_only=True);records=[]
    for entry in load(BASE/'train/report.json')['cases']:
        name=entry['image'];path=GEOMETRY/'full_train'/(Path(name).stem+'_predictions.json');pins[str(path)]=sha(path);current=load(path)['trial']
        indices=[i for i,r in enumerate(metadata) if r['image']==name and r['kind']=='novel_weak_proposal']
        native=[metadata[i]['proposal'] for i in indices]
        with torch.inference_mode():scores=head(data['features'][indices]).softmax(dim=1).tolist()
        trial=select(current,native,scores,trained['checkpoint']['sha256'])
        save(destination/(Path(name).stem+'_predictions.json'),dict(image=name,current=current,trial=trial,proposals=native,probabilities=scores))
        targets=read_targets('train',name,[2736,3648],entry['label_sha256'],pins)
        oh,nh=matches(current['all_predictions'],targets)[0],matches(trial['all_predictions'],targets)[0]
        records.append(dict(image=name,current=metric(current['all_predictions'],targets),trial=metric(trial['all_predictions'],targets),
            gained=sorted(nh-oh),lost=sorted(oh-nh),additions=len(trial['paired_semantic_additions'])))
    totals={kind:{key:sum(r[kind][key] for r in records) for key in ('tp','unmatched','fn','predictions','targets')} for kind in ('current','trial')}
    assert totals['current']==load(GEOMETRY/'full_train/report.json')['summary']['trial']
    normal=sum(r['trial']['predictions'] for r in records if r['image'].startswith('normal_'))
    qualifies=totals['trial']['tp']>totals['current']['tp'] and totals['trial']['unmatched']<=totals['current']['unmatched'] and normal==0 and not any(r['lost'] for r in records)
    assert {p:sha(Path(p)) for p in pins}==pins and paired_runtime_fingerprint(REPO)==frozen
    report=dict(status='complete',qualifies=qualifies,summary=totals,cases=records,pins=pins,normal_cues=normal,
        exact_head_full_not_OOF=True,no_deployment=True,field_accuracy=False)
    save(destination/'report.json',report);print(str(dict(qualifies=qualifies,summary=totals)),flush=True)

if __name__=='__main__':main()
