"""Post-hoc ALL-TRAIN acquisition funnel; never used by the inference policy."""
import collections
import sys
from pathlib import Path
sys.dont_write_bytecode = True
from prepare_paired_port_semantics import ROOT, REPO, load, save, sha
from current_port_baseline_audit import BASE, read_targets
from audit_port_multiscale_acceptance import matches, metric, overlap
from inspection_agent.paired_native_pose import native_pose_runtime_fingerprint

SOURCE = ROOT / 'artifacts/evidence_first_resolution_20261004'
AUDIT = ROOT / 'artifacts/evidence_first_resolution_replay_20261004/report.json'
OUT = ROOT / 'artifacts/evidence_first_funnel_20261004'


def covering(rows, target):
    return [i for i, row in enumerate(rows)
            if row['class_id'] == target['class_id'] and overlap(row['box_xyxy'], target['box']) >= .5]


def main():
    if OUT.exists():
        raise FileExistsError('Preserve complete diagnosis')
    report = load(SOURCE / 'report.json')
    audit = load(AUDIT)
    assert audit['status'] == 'pass'
    assert audit['source_report_sha256'] == sha(SOURCE / 'report.json')
    assert all(sha(Path(p)) == v for p, v in report['pins'].items())
    assert native_pose_runtime_fingerprint(REPO) == report['runtime']
    pins = {str(p): sha(p) for p in (Path(__file__), SOURCE / 'report.json', AUDIT)}
    stages = collections.Counter()
    reasons = collections.Counter()
    unmatched_kinds = collections.Counter()
    rows = []
    for entry in load(BASE / 'train/report.json')['cases']:
        name = entry['image']
        path = SOURCE / 'train' / (Path(name).stem + '_predictions.json')
        pins[str(path)] = sha(path)
        case = load(path)
        targets = read_targets('train', name, [2736, 3648], entry['label_sha256'], pins)
        old, new = case['current']['all_predictions'], case['trial']['all_predictions']
        prior, _, _ = matches(old, targets)
        after, unmatched, _ = matches(new, targets)
        details = []
        for ti, target in enumerate(targets):
            if ti in prior:
                continue
            seed = covering(case['seeds'], target)
            action = covering(case['chosen_seeds'], target)
            proposal = covering(case['proposals'], target)
            semantic = [j for j in proposal if
                        max(range(3), key=lambda c: case['probabilities'][j][c]) == target['class_id'] + 1
                        and case['probabilities'][j][target['class_id'] + 1] >= .98]
            flags = dict(missed_by_prefix=True, two_vote_seed=bool(seed),
                         selected_action=bool(action), final_three_vote_proposal=bool(proposal),
                         semantic_gate=bool(semantic), selected_gain=ti in after)
            stages.update({k: int(v) for k, v in flags.items()})
            # These sets need not be nested: ROI around a neighbour can recover
            # this target. Diagnose final coverage first, not an invented chain.
            reason = ('gained' if ti in after else 'selection_or_budget' if semantic else
                      'semantic_rejection' if proposal else 'no_final_proposal_despite_action' if action else
                      'seed_not_chosen' if seed else 'no_two_vote_seed')
            reasons[reason] += 1
            details.append(dict(target_index=ti, **flags, reason=reason))
        added_unmatched = []
        for pi in sorted(unmatched):
            if pi < len(old):
                continue
            p = new[pi]
            best = max((overlap(p['box_xyxy'], t['box']) for t in targets
                        if p['class_id'] == t['class_id']), default=0.)
            kind = 'same_class_duplicate_target_match' if best >= .5 else 'below_IoU_0.5_or_unlabelled'
            unmatched_kinds[kind] += 1
            added_unmatched.append(dict(prediction_index=pi, maximum_same_class_GT_IoU=best, kind=kind))
        rows.append(dict(image=name, current=metric(old, targets), trial=metric(new, targets),
                         missed_targets=details, added_unmatched=added_unmatched))
    assert len(rows) == 192 and sum(reasons.values()) == stages['missed_by_prefix'] == 46
    for version in ('current', 'trial'):
        totals = {k: sum(r[version][k] for r in rows) for k in ('tp', 'unmatched', 'fn', 'predictions', 'targets')}
        assert totals == report['summary']['summary'][version]
    assert all(sha(Path(p)) == v for p, v in pins.items())
    OUT.mkdir()
    save(OUT / 'report.json', dict(status='complete', source_report_sha256=sha(SOURCE / 'report.json'),
         posthoc_all_TRAIN_only=True, no_inference_policy_changes=True, no_validation_read=True,
         stage_counts_not_necessarily_nested=dict(stages), exclusive_reasons=dict(reasons),
         added_unmatched_categories=dict(unmatched_kinds), cases=rows, pins=pins,
         unmatched_not_confirmed_physical_false_alarm=True, field_accuracy=False))
    print(dict(stages=stages, reasons=reasons, added_unmatched=unmatched_kinds), flush=True)


if __name__ == '__main__':
    main()
