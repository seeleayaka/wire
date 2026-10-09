"""Post-selection fixed-budget oracle, never a runtime GT rule or achieved gain."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,load,save,sha


def budget_ceiling(targets,hits,current_count,primary_count):
    if any(type(v) is not int or v<0 for v in (targets,hits,current_count,primary_count)):
        raise ValueError('nonnegative exact integer counts required')
    if primary_count>5 or primary_count>current_count or hits>targets or hits>current_count or current_count>primary_count+5:
        raise ValueError('invalid protected 5+5 count contract')
    room=primary_count+5-current_count;misses=targets-hits
    return dict(capacity=primary_count+5,remaining_slots=room,missing_targets=misses,
                ideal_new_hits_upper_bound=min(room,misses),unavoidable_misses_lower_bound=max(0,misses-room))


def main():
    source=ROOT/'artifacts/allport480_source_20261005';auditpath=ROOT/'artifacts/allport480_source_audit_20261005/report.json'
    out=ROOT/'artifacts/allport480_fixed_capacity_audit_20261005'
    if out.exists():raise FileExistsError('preserve mathematical ceiling audit')
    report=load(source/'report.json');audit=load(auditpath)
    if audit['status']!='pass' or audit['source_report_sha256']!=sha(source/'report.json') or len(audit['source_case_sha256'])!=192:raise ValueError('complete independent source required')
    pins={str(p):sha(p) for p in (Path(__file__),source/'report.json',auditpath)};rows=[]
    for entry in report['cases']:
        name=entry['image'];path=source/'train'/(Path(name).stem+'_predictions.json');pins[str(path)]=sha(path)
        if pins[str(path)]!=audit['source_case_sha256'][str(path)]:raise ValueError('audited selection changed')
        case=load(path);current=entry['current'];selection=case['current']
        if len(selection['all_predictions'])!=current['predictions']:raise ValueError('protected prefix count mismatch')
        rows.append(dict(image=name,current_hits=current['tp'],**budget_ceiling(current['targets'],current['tp'],current['predictions'],len(selection['primary']))))
    added=sum(r['ideal_new_hits_upper_bound'] for r in rows);blocked=sum(r['unavoidable_misses_lower_bound'] for r in rows)
    targets=report['summary']['current']['targets'];existing=report['summary']['current']['tp']
    if added+blocked!=targets-existing or any(sha(p)!=d for p,d in pins.items()):raise ValueError('ceiling evidence/count drift')
    out.mkdir();save(out/'report.json',dict(status='complete_post_selection_upper_bound',sources=192,targets=targets,protected_research_hits=existing,
        mathematically_maximum_hits_with_frozen_prefix_and_budget=existing+added,potential_additional_hits_upper_bound=added,
        unavoidable_GT_misses_lower_bound=blocked,cases=rows,pins=pins,no_model_inference=True,no_GT_runtime_inputs=True,
        no_budget_changes=True,no_deployment=True,achieved_new_hits=None,field_accuracy=None,
        warning='Ideal mathematical ceiling only, not achievable/observed accuracy. Keep protected5+5 and both old prefixes unchanged.'))
    print(dict(status='complete_post_selection_upper_bound',existing=existing,mathematical_ceiling=existing+added,blocked_by_frozen_budget=blocked,potential_upper_bound=added))


if __name__=='__main__':main()
