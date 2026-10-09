"""One-shot posthoc aggregation; preserve failed progress and every inference byte.

Fix only the wrong original277-vs-accepted295 assertion. No inference, GT edits,
selection/gate changes, recovery pretending to be original process completion.
"""
import copy
import subprocess
import sys
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,REPO,DATA,load,save,sha
from inspection_agent.paired_native_pose import native_pose_runtime_fingerprint
from current_port_baseline_audit import BASE,read_targets
from audit_allport480_source import score
from protected_allport_baseline import check_accepted,check_totals

SOURCE=ROOT/'artifacts/allport480_source_20261005'
ACCEPTED=ROOT/'artifacts/paired_pose_native_three_20261004'
RESEARCH=ROOT/'artifacts/fine_consensus_rank_20261004'


def main():
    if (SOURCE/'report.json').exists():raise FileExistsError('never overwrite a final summary')
    protocol=load(SOURCE/'protocol.json');failed=load(SOURCE/'progress.json');partial=load(SOURCE/'partial_metrics.json')
    if failed['status']!='failed' or failed['completed']!=192 or failed['error']!='ValueError: protected baseline metrics mismatch':
        raise ValueError('only this fully collected aggregation failure can be recovered')
    if partial['completed']!=192 or partial['total']!=192 or len(partial['cases'])!=192:raise ValueError('incomplete saved collection')
    dirty=lambda:subprocess.check_output(['E:/Git/cmd/git.exe','-C',str(REPO),'status','--porcelain'],text=True)
    if native_pose_runtime_fingerprint(REPO)!=protocol['runtime'] or dirty()!=protocol['mainline_dirty_before']:raise ValueError('E mainline changed')
    pins=copy.deepcopy(protocol['pins'])
    if any(sha(p)!=d for p,d in pins.items()):raise ValueError('initial collection provenance changed')
    prior=load(RESEARCH/'report.json');acceptedreport=load(ACCEPTED/'report.json')
    if any(sha(p)!=d for p,d in prior['pins'].items()) or any(sha(p)!=d for p,d in acceptedreport['pins'].items()):raise ValueError('protected ancestral evidence changed')
    for p in [Path(__file__),Path(__file__).with_name('protected_allport_baseline.py'),SOURCE/'protocol.json',SOURCE/'progress.json',SOURCE/'partial_metrics.json',ACCEPTED/'report.json']:
        pins[str(p)]=sha(p)
    entries=load(BASE/'train/report.json')['cases'];indexed={r['image']:r for r in entries}
    names=protocol['train_sources']
    if len(set(names))!=192 or set(indexed)!=set(names) or {r['image'] for r in partial['cases']}!=set(names):raise ValueError('source membership differs')
    if {p.name for p in (SOURCE/'train').glob('*_predictions.json')}!={Path(n).stem+'_predictions.json' for n in names}:raise ValueError('saved output inventory differs')
    rows=[];acceptedrows=[];outputpins={};proposals=0;fresh_calls=0
    for olditem in partial['cases']:
        name=olditem['image'];stem=Path(name).stem;path=SOURCE/'train'/(stem+'_predictions.json');outputpins[str(path)]=sha(path);case=load(path)
        source=DATA/'images/train01'/name
        if case['image']!=name or sha(source)!=case['source_sha256'] or case['roles']!=protocol['roles']:raise ValueError('source output identity changed')
        pins[str(source)]=case['source_sha256']
        legacy_path=BASE/'train'/(stem+'_predictions.json');research_path=RESEARCH/'train'/(stem+'_predictions.json');accepted_path=ACCEPTED/'full_train'/(stem+'_predictions.json')
        for p in [legacy_path,research_path,accepted_path]:pins[str(p)]=sha(p)
        legacy=load(legacy_path)['trial'];research=load(research_path);accepted=load(accepted_path)['trial']
        if case['original']!=legacy or case['current']!=research['trial'] or research['current']!=accepted:raise ValueError('wrong protected generation')
        check_accepted(legacy,accepted,case['current'],case['trial'])
        targets=read_targets('train',name,[2736,3648],indexed[name]['label_sha256'],pins)
        metrics={};hits={}
        for version in ('original','current','trial'):metrics[version],hits[version]=score(case[version]['all_predictions'],targets)
        row=dict(image=name,**metrics,gained=sorted(hits['trial']-hits['current']),lost=sorted(hits['current']-hits['trial']),lost_original=sorted(hits['original']-hits['trial']),skip_reason=case['skip_reason'])
        if row!=olditem:raise ValueError('saved full-case collection scores changed')
        acceptedmetric,acceptedhits=score(accepted['all_predictions'],targets)
        acceptedrows.append(dict(image=name,metrics=acceptedmetric,lost=sorted(acceptedhits-hits['trial']),path=str(accepted_path),sha256=pins[str(accepted_path)]))
        rows.append(row);proposals+=len(case['proposals'])
        fresh_calls+=sum(v.get('view','').startswith('fresh_') for v in case['new_voter_views'])
    totals={v:{k:sum(r[v][k] for r in rows) for k in ('tp','unmatched','fn','predictions','targets')} for v in ('original','current','trial')}
    acceptedtotal={k:sum(r['metrics'][k] for r in acceptedrows) for k in ('tp','unmatched','fn','predictions','targets')}
    check_totals(dict(**totals,accepted=acceptedtotal),legacy=True)
    normal=sum(r['trial']['predictions'] for r in rows if r['image'].startswith('normal_'))
    qualifies=totals['trial']['tp']>298 and totals['trial']['unmatched']<=4 and normal==0 and not any(r['lost'] or r['lost_original'] for r in rows) and not any(r['lost'] for r in acceptedrows)
    if any(sha(p)!=d for p,d in pins.items()) or any(sha(p)!=d for p,d in outputpins.items()) or native_pose_runtime_fingerprint(REPO)!=protocol['runtime'] or dirty()!=protocol['mainline_dirty_before']:raise ValueError('evidence changed during summary recovery')
    result=dict(status='source_pass_requires_independent_replay_and_fresh_holdouts' if qualifies else 'rejected',qualifies=qualifies,summary=totals,cases=rows,normal_cues=normal,pins=pins,runtime=protocol['runtime'],
        seconds=failed['seconds'],fresh_view_calls=fresh_calls,proposals=proposals,no_heldout_reads=True,no_deployment=True,field_accuracy=None,mainline_unchanged=True,
        aggregation_recovery=dict(original_process_status='failed_after_192_saved_sources',original_error=failed['error'],original_progress_preserved=True,original_baseline='legacy_resolution277_not_accepted295',accepted_baseline='paired_pose_native_three_full_train295',accepted_summary=acceptedtotal,accepted_cases=acceptedrows,saved_source_case_sha256=outputpins,no_new_model_inference=True,no_output_edits=True,no_gate_changes=True,requires_independent_replay=True))
    save(SOURCE/'report.json',result)
    save(SOURCE/'summary_recovery_progress.json',dict(status='posthoc_complete_requires_independent_audit',qualifies=qualifies,original_worker_status='failed',summary=totals,accepted_summary=acceptedtotal))
    print(dict(status=result['status'],summary=totals,accepted_summary=acceptedtotal,normal_cues=normal,fresh_view_calls=fresh_calls))


if __name__=='__main__':main()
