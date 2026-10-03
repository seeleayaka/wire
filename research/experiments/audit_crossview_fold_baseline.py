"""Same classifier fold on both sides; cannot substitute for deployment."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,load,save,sha
from prepare_paired_semantic_geometry import NEW as GEOMETRY
from current_port_baseline_audit import BASE,read_targets
from paired_crossview_selection import select
from audit_port_multiscale_acceptance import metric,matches
OUT=ROOT/'artifacts/paired_crossview_fold_baseline_20261003'
CROSS=ROOT/'artifacts/paired_crossview_20261003'


def main():
    if OUT.exists():raise FileExistsError('Preserve paired baseline audit')
    OUT.mkdir();pins={str(p):sha(p) for p in (Path(__file__),Path(__file__).with_name('paired_crossview_selection.py'),
        ROOT/'artifacts/paired_crossview_fold_baseline_preregistration_20261003/PLAN.md',CROSS/'source_train_oof/report.json',
        GEOMETRY/'source_train_oof/report.json')};records=[]
    for entry in load(BASE/'train/report.json')['cases']:
        name=entry['image'];prefix=GEOMETRY/'source_train_oof'/(Path(name).stem+'_predictions.json');path=CROSS/'source_train_oof'/(Path(name).stem+'_predictions.json')
        pins[str(path)]=sha(path);pins[str(prefix)]=sha(prefix);baseline=load(prefix);current=baseline['trial'];cross=load(path)
        fold=next((r['fold'] for r in load(GEOMETRY/'features_train/samples.json') if r['image']==name),0)
        context_path=ROOT/'artifacts/paired_port_semantics_20261003/heads_oof'/f'fold{fold}_head.pt'
        footprint_path=ROOT/'artifacts/paired_boxpool_original_rows_20261003/heads_oof'/f'fold{fold}_head.pt'
        heads=[sha(context_path),sha(footprint_path)];pins[str(context_path)]=heads[0];pins[str(footprint_path)]=heads[1]
        trial=select(current,cross['proposals'],cross['member_probabilities'],heads)
        save(OUT/(Path(name).stem+'_predictions.json'),dict(image=name,current=current,trial=trial))
        targets=read_targets('train',name,[2736,3648],entry['label_sha256'],pins);oh,nh=matches(current['all_predictions'],targets)[0],matches(trial['all_predictions'],targets)[0]
        records.append(dict(image=name,current=metric(current['all_predictions'],targets),trial=metric(trial['all_predictions'],targets),
            gained=sorted(nh-oh),lost=sorted(oh-nh),additions=len(trial['paired_crossview_additions'])))
    totals={kind:{k:sum(r[kind][k] for r in records) for k in ('tp','unmatched','fn','predictions','targets')} for kind in ('current','trial')}
    assert totals['current']==load(GEOMETRY/'source_train_oof/report.json')['summary']['trial']
    normal=sum(r['trial']['predictions'] for r in records if r['image'].startswith('normal_'))
    qualifies=totals['trial']['tp']>totals['current']['tp'] and totals['trial']['unmatched']<=totals['current']['unmatched'] and not any(r['lost'] for r in records) and normal==0
    assert {p:sha(Path(p)) for p in pins}==pins
    save(OUT/'report.json',dict(status='complete',qualifies_fold_diagnostic=qualifies,summary=totals,cases=records,pins=pins,
        deployed_baseline_remains286=True,not_deployment_acceptance=True,detectors_not_OOF=True,no_deployment=True,field_accuracy=False))
    print(str(dict(qualifies=qualifies,summary=totals)),flush=True)

if __name__=='__main__':main()
