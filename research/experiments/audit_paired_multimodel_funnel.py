"""Explain saved classifier rejection; no inference or threshold selection."""
import collections,sys
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,load,save,sha
OUT=ROOT/'artifacts/paired_multimodel_funnel_20261003'
SCORED=ROOT/'artifacts/paired_multimodel_head_20261003'


def main():
    if OUT.exists():raise FileExistsError('Preserve funnel audit')
    report=load(SCORED/'train/report.json');assert report['status']=='complete' and not report['qualifies']
    reasons=collections.Counter();classes=collections.Counter();per_source=[];pins={str(SCORED/'train/report.json'):sha(SCORED/'train/report.json')}
    for record in report['cases']:
        path=SCORED/'train'/(Path(record['image']).stem+'_predictions.json');pins[str(path)]=sha(path);row=load(path);counts=collections.Counter()
        for proposal,prob in zip(row['proposals'],row['probabilities']):
            cls=max(range(3),key=lambda i:prob[i]);classes[cls]+=1
            reason='other_class' if cls==0 else 'below_fixed98' if prob[cls]<.98 else 'class_disagrees_with_witness' if cls!=proposal['class_id']+1 else 'eligible_before_budget_and_overlap'
            counts[reason]+=1;reasons[reason]+=1
        per_source.append(dict(image=record['image'],reasons=dict(counts),proposals=len(row['proposals']),
            feature_status=row['feature_status'],accepted=record['additions']))
    assert sum(reasons.values())==report['fresh_pairs']+report['reused_pairs']
    OUT.mkdir();save(OUT/'report.json',dict(status='complete',source_count=192,counts=dict(reasons),classes=dict(classes),
        sources=per_source,pins=pins,GT_not_used=True,no_score_change=True,no_deployment=True))
    print(str(dict(reasons)),flush=True)

if __name__=='__main__':main()
