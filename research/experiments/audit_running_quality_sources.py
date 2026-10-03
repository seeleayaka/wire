"""Read completed source outputs; prove early impossibility without mutation."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
from current_port_baseline_audit import ROOT,BASE,load,save,read_targets
from inspection_agent.optional_port_crop_review import sha
from audit_port_multiscale_acceptance import metric
from audit_port_matching_cardinality import maximum_matching
OUT=ROOT/'artifacts/dense_quality_20261003/source_acceptance'


def main():
    progress=load(OUT/'progress.json')
    if progress['status']!='running':print(str(progress));return
    stage=progress['stage'];done=progress['completed'];folder=OUT/stage
    destination=OUT/f'{stage}_partial_bound_{done:03d}.json'
    if destination.exists():print(str(load(destination)['summary']));return
    entries=load(BASE/stage/'report.json')['cases'];pins={};records=[]
    totals={v:{k:0 for k in ('tp','unmatched','fn','predictions','targets')} for v in ('current','trial')}
    for entry in entries[:done]:
        path=folder/(Path(entry['image']).stem+'_predictions.json');pins[str(path)]=sha(path);case=load(path)
        targets=read_targets(stage,entry['image'],case['raw']['source_shape'],entry['label_sha256'],pins)
        old,new=case['current']['all_predictions'],case['trial']['all_predictions']
        assert new[:len(old)]==old
        om,nm=metric(old,targets),metric(new,targets);oldmax=len(maximum_matching(old,targets));newmax=len(maximum_matching(new,targets))
        assert oldmax==om['tp'] and om==entry['metrics']['trial']
        old_unmatched_lower_bound=len(old)-oldmax;new_unmatched_lower_bound=len(new)-newmax
        assert new_unmatched_lower_bound>=old_unmatched_lower_bound
        for v,m in (('current',om),('trial',nm)):
            for k,value in m.items():totals[v][k]+=value
        if len(new)>len(old):records.append(dict(image=entry['image'],current=om,trial=nm,
            old_maximum_tp=oldmax,new_maximum_tp=newmax,
            unmatched_lower_bound_increase=new_unmatched_lower_bound-old_unmatched_lower_bound))
    impossible=any(r['unmatched_lower_bound_increase']>0 for r in records)
    result=dict(status='complete',completed_sources_only=done,planned_sources=len(entries),stage=stage,summary=totals,
        cases_with_additions=records,reject_proof=impossible,
        theorem='Adding k prediction vertices can increase maximum matched targets by at most k; old current greedy equals maximum, so append-only unmatched count cannot decrease on any source',
        complete_source_acceptance_not_claimed=True,does_not_stop_or_modify_running_job=True,pins=pins,field_accuracy=False)
    save(destination,result);print(str({k:v for k,v in result.items() if k not in ('pins','theorem')}),flush=True)


if __name__=='__main__':main()
