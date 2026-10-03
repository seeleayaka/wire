"""Independent score-gate and unchanged weak-label accounting for precision trial."""
from pathlib import Path
from prepare_paired_port_semantics import ROOT,REPO,load,save,sha
from current_port_baseline_audit import BASE,read_targets
from audit_port_multiscale_acceptance import metric,matches
from inspection_agent.paired_native_pose import native_pose_runtime_fingerprint
SOURCE=ROOT/'artifacts/native_negative_verification_20261004'
OUT=ROOT/'artifacts/native_negative_verification_audit_20261004'

def main():
    if OUT.exists():raise FileExistsError('Preserve independent precision replay')
    report=load(SOURCE/'report.json');assert report['status']=='rejected'
    assert all(sha(Path(p))==v for p,v in report['pins'].items()) and native_pose_runtime_fingerprint(REPO)==report['runtime']
    entries={r['image']:r for r in load(BASE/'train/report.json')['cases']};pins={};counts={};removed=0
    for stage,summary in report['stages'].items():
        rows=load(SOURCE/stage/'report.json')['cases']
        for row in rows:
            case=load(SOURCE/stage/(Path(row['image']).stem+'_predictions.json'))
            assert len(case['valid_context_indices'])==len(case['probabilities'])
            reject={index for index,prob in zip(case['valid_context_indices'],case['probabilities']) if prob[0]>=.98}
            assert sorted(reject)==case['removed_indices'] and case['trial']==[p for i,p in enumerate(case['current']) if i not in reject]
            targets=read_targets('train',row['image'],[2736,3648],entries[row['image']]['label_sha256'],pins)
            for version in ('current','trial'):assert metric(case[version],targets)==row[version]
            old=matches(case['current'],targets)[0];new=matches(case['trial'],targets)[0];assert row['lost']==sorted(old-new)
            removed+=len(reject)
        totals={version:{k:sum(row[version][k] for row in rows) for k in ('tp','unmatched','fn','predictions','targets')} for version in ('current','trial')}
        assert totals==summary['summary'];counts[stage]=len(rows)
    assert all(sha(Path(p))==v for p,v in report['pins'].items()) and native_pose_runtime_fingerprint(REPO)==report['runtime']
    OUT.mkdir();result=dict(status='complete',counts=counts,removed=removed,exact_cached_gate_and_independent_GT_scoring=True,
        unchanged_runtime=True,no_deployment=True,field_accuracy=False,source_report_sha256=sha(SOURCE/'report.json'))
    save(OUT/'report.json',result);print(result)

if __name__=='__main__':main()
