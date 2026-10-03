"""Independent bounded cached replay of completed raw pose experiment."""
from pathlib import Path
from prepare_paired_port_semantics import ROOT,REPO,load,save,sha
from current_port_baseline_audit import BASE,read_targets
from audit_port_multiscale_acceptance import metric,matches
from inspection_agent.paired_native_pose import native_pose_runtime_fingerprint,HEAD_SHA
from raw_pose_consensus import select
SOURCE=ROOT/'artifacts/raw_pose_consensus_20261004'
OUT=ROOT/'artifacts/raw_pose_consensus_audit_20261004'

def main():
    if OUT.exists():raise FileExistsError('Preserve independent replay')
    report=load(SOURCE/'report.json');assert report['status'] in ('rejected','source_pass_requires_actual_reference_ROI_gates')
    pins=report['pins'];assert all(sha(Path(p))==v for p,v in pins.items())
    assert native_pose_runtime_fingerprint(REPO)==report['runtime']
    OUT.mkdir();counts={};added=0;replay_pins={}
    for stage,summary in report['stages'].items():
        rows=load(SOURCE/stage/'report.json')['cases'];entries={r['image']:r for r in load(BASE/stage/'report.json')['cases']}
        for row in rows:
            path=SOURCE/stage/(Path(row['image']).stem+'_predictions.json');case=load(path)
            assert select(case['current'],case['proposals'],case['probabilities'],HEAD_SHA)==case['trial']
            assert case['trial']['all_predictions'][:len(case['current']['all_predictions'])]==case['current']['all_predictions']
            targets=read_targets(stage,row['image'],[2736,3648],entries[row['image']]['label_sha256'],replay_pins)
            for version in ('current','trial'):assert metric(case[version]['all_predictions'],targets)==row[version]
            old=matches(case['current']['all_predictions'],targets)[0];new=matches(case['trial']['all_predictions'],targets)[0]
            assert sorted(new-old)==row['gained'] and sorted(old-new)==row['lost']
            assert len(case['trial']['primary'])<=5 and len(case['trial']['all_predictions'])<=len(case['trial']['primary'])+5
            for proposal in case['proposals']:assert len(set(proposal['semantic_model_vote_sha256']))>=3
            added+=len(case['trial']['all_predictions'])-len(case['current']['all_predictions'])
        totals={version:{k:sum(row[version][k] for row in rows) for k in ('tp','unmatched','fn','predictions','targets')} for version in ('current','trial')}
        assert totals==summary['summary'];counts[stage]=len(rows)
    assert all(sha(Path(p))==v for p,v in pins.items()) and all(sha(Path(p))==v for p,v in replay_pins.items())
    assert native_pose_runtime_fingerprint(REPO)==report['runtime']
    result=dict(status='complete',experiment_status=report['status'],counts=counts,added=added,
        source_report_sha256=sha(SOURCE/'report.json'),independent_scoring=True,exact_cached_selector_replay=True,
        unchanged_runtime=True,not_actual_reference_ROI_or_Qt_acceptance=True,field_accuracy=False)
    save(OUT/'report.json',result);print(result)

if __name__=='__main__':main()
