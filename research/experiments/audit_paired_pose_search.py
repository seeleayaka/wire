"""Independent saved-prediction audit and post-selection miss funnel."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
from prepare_paired_port_semantics import ROOT, REPO, load, save, sha
from current_port_baseline_audit import BASE, read_targets
from audit_port_multiscale_acceptance import metric, matches, overlap
from paired_pose_search import select
from inspection_agent.paired_port_geometry import HEAD_SHA
from inspection_agent.paired_median_geometry import median_runtime_fingerprint
from run_paired_pose_search import OUT as TRIAL, MEDIAN
OUT = ROOT / 'artifacts/paired_pose_search_audit_20261003'


def main():
    if OUT.exists(): raise FileExistsError('Preserve independent pose audit')
    report = load(TRIAL / 'report.json'); frozen = median_runtime_fingerprint(REPO)
    assert frozen == report['median_runtime_fingerprint']
    pins = dict(report['pins']); assert {p: sha(Path(p)) for p in pins} == pins
    for path in (Path(__file__), TRIAL / 'report.json', TRIAL / 'protocol.json'):
        pins[str(path)] = sha(path)
    rows = []; totals = {}; funnel = {}; per_case_fp_increase = []
    for stage in report['stages']:
        source_entries = load(BASE / stage / 'report.json')['cases']
        stage_rows = []; stage_funnel = {}
        for entry in source_entries:
            name = entry['image']; path = TRIAL / stage / (Path(name).stem + '_predictions.json')
            pins[str(path)] = sha(path); case = load(path)
            current_path = MEDIAN / stage / path.name; pins[str(current_path)] = sha(current_path)
            assert case['current'] == load(current_path)['trial']
            trial = select(case['current'], case['proposals'], case['probabilities'], HEAD_SHA)
            assert trial == case['trial']
            additions = trial['paired_semantic_additions']
            assert len({p['pose_parent_seed_id'] for p in additions}) == len(additions)
            old, new = case['current']['all_predictions'], trial['all_predictions']
            assert new[:len(old)] == old
            targets = read_targets(stage, name, [2736, 3648], entry['label_sha256'], pins)
            oh, nh = matches(old, targets)[0], matches(new, targets)[0]
            row = dict(image=name, current=metric(old, targets), trial=metric(new, targets),
                gained=sorted(nh-oh), lost=sorted(oh-nh), additions=len(additions), misses=[])
            if row['trial']['unmatched'] > row['current']['unmatched']:
                per_case_fp_increase.append(dict(stage=stage, image=name))
            if name.startswith('normal_'): assert not new
            parent_full = len(old) - len(case['current']['primary']) == 5
            for ti in sorted(set(range(len(targets))) - nh):
                target = targets[ti]
                relevant = [(p, v) for p, v in zip(case['proposals'], case['probabilities'])
                    if p['class_id'] == target['class_id'] and overlap(p['box_xyxy'], target['box']) >= .5]
                accepted = [(p, v) for p, v in relevant if max(range(3), key=lambda i: v[i]) == target['class_id'] + 1 and v[target['class_id'] + 1] >= .98]
                reason = 'current_shared_budget_full' if parent_full else 'no_valid_pose_geometry' if not relevant else 'below_unchanged_semantic_gate' if not accepted else 'parent_or_global_selection'
                stage_funnel[reason] = stage_funnel.get(reason, 0) + 1
                row['misses'].append(dict(target_index=ti, reason=reason,
                    poses=len(relevant), best_raw_probability=max((v[target['class_id']+1] for p,v in relevant), default=None)))
            stage_rows.append(row); rows.append(dict(stage=stage, **row))
        measured = {kind: {k: sum(r[kind][k] for r in stage_rows) for k in ('tp', 'unmatched', 'fn', 'predictions', 'targets')} for kind in ('current', 'trial')}
        assert measured == report['stages'][stage]['summary']
        totals[stage] = measured; funnel[stage] = stage_funnel
    assert {p: sha(Path(p)) for p in pins} == pins
    assert median_runtime_fingerprint(REPO) == frozen
    OUT.mkdir(); result = dict(status='complete', original_trial_status=report['status'], stages=totals,
        miss_funnel=funnel, per_case_unmatched_increases=per_case_fp_increase, cases=rows, pins=pins,
        independent_selection_scoring=True, GT_only_post_selection=True, no_deployment=True, field_accuracy=False)
    save(OUT / 'report.json', result)
    print(str(dict(stages=totals, miss_funnel=funnel, per_case_unmatched_increases=per_case_fp_increase)), flush=True)


if __name__ == '__main__': main()
