"""Exact rejection-only oracle of fixed source candidates, not a runtime rule."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
from current_port_baseline_audit import ROOT,REPO,load,save,read_targets
from inspection_agent.optional_port_crop_review import sha
from inspection_agent.resolution_loose_plug_support import resolution_runtime_fingerprint
from audit_port_matching_cardinality import maximum_matching
from audit_port_multiscale_acceptance import metric
OUT=ROOT/'artifacts/paired_support_graph_20261003'


def main():
    path=OUT/'candidate_set_upper_bound.json'
    if path.exists():raise FileExistsError('Preserve prior upper bound')
    protocol=load(OUT/'source_selections/protocol.json');assert protocol['status']=='complete'
    pins=dict(protocol['pins']);frozen=resolution_runtime_fingerprint(REPO);assert frozen==protocol['runtime_fingerprint']
    stages={};rows=[]
    for stage in ('train','inner','outer'):
        totals=dict(current_tp=0,candidate_maximum_tp=0,targets=0,current_unmatched=0,all_candidates_unmatched=0)
        for record in load(OUT/'source_selections'/stage/'index.json')['records']:
            p=Path(record['path']);assert sha(p)==record['sha256'];pins[str(p)]=record['sha256'];case=load(p)
            targets=read_targets(stage,record['image'],case['teacher']['predictions']['source_shape'],case['entry']['label_sha256'],pins)
            old,new=case['current']['all_predictions'],case['trial']['all_predictions']
            baseline=metric(old,targets);assert len(maximum_matching(old,targets))==baseline['tp']
            upper=len(maximum_matching(new,targets))
            row=dict(stage=stage,image=record['image'],current_tp=baseline['tp'],candidate_maximum_tp=upper,targets=len(targets),
                     current_unmatched=baseline['unmatched'],all_candidates_unmatched=metric(new,targets)['unmatched'])
            for k in totals:totals[k]+=row[k]
            if record['additions']:rows.append(row)
        stages[stage]=dict(totals,possible_gain=totals['candidate_maximum_tp']-totals['current_tp'])
    impossible=any(stages[s]['possible_gain']==0 for s in ('train','inner'))
    assert {p:sha(Path(p)) for p in pins}==pins and resolution_runtime_fingerprint(REPO)==frozen
    result=dict(status='complete',summary=stages,cases_with_additions=rows,pins=pins,
        rejects_required_net_gain_before_expensive_reference=impossible,
        reference_filter_cannot_create_native_boxes_or_modify_native_geometry=True,
        rejection_only_oracle_never_candidate_selection=True,no_reference_performance_claim=True,
        no_rule_or_threshold_changed=True,field_accuracy=False)
    save(path,result)
    if impossible:save(OUT/'pipeline_progress.json',dict(status='rejected_no_possible_required_inner_gain',upper_bound=str(path),fresh_reference_not_run=True))
    print(str(dict(summary=stages,rejects_required_net_gain=impossible)),flush=True)


if __name__=='__main__':main()
