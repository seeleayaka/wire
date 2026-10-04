"""All-source posthoc ROI coverage and missing-checkpoint routing analysis.

Ground truth is diagnostic only. This module is not imported by inference.
"""
import argparse
import collections
import sys
from pathlib import Path
sys.dont_write_bytecode = True
from prepare_paired_port_semantics import ROOT, REPO, load, save, sha
from current_port_baseline_audit import BASE, read_targets
from audit_port_multiscale_acceptance import matches, overlap
from unresolved_voter_seeds import pool_rows
from inspection_agent.paired_native_pose import native_pose_runtime_fingerprint

PREP = ROOT / 'artifacts/two_vote_resolution_preparation_20261004'


def target_fraction(box, window):
    l, t, r, b = box
    a, c, d, e = window
    area = max(0., min(r, d)-max(l, a)) * max(0., min(b, e)-max(t, c))
    return area / ((r-l)*(b-t))


def vote_map(models, source, target):
    pool = pool_rows(models, source, (2736, 3648))
    best = {m['weight_sha256']: 0. for m in models}
    for weight, row in pool:
        if row['class_id'] == target['class_id']:
            best[weight] = max(best[weight], overlap(row['box_xyxy'], target['box']))
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', required=True, choices=['evidence_first', 'zoom2'])
    args = parser.parse_args()
    name = 'evidence_first_resolution' if args.experiment == 'evidence_first' else 'zoom2_resolution'
    source = ROOT / ('artifacts/' + name + '_20261004')
    auditpath = ROOT / ('artifacts/' + name + '_replay_20261004/report.json')
    out = ROOT / ('artifacts/' + name + '_coverage_20261004')
    if out.exists():
        raise FileExistsError('Preserve diagnostic evidence')
    report = load(source / 'report.json')
    audit = load(auditpath)
    assert audit['status'] == 'pass' and audit['source_report_sha256'] == sha(source / 'report.json')
    assert all(sha(Path(p)) == v for p, v in report['pins'].items())
    assert native_pose_runtime_fingerprint(REPO) == report['runtime']
    pins = {str(p): sha(p) for p in (Path(__file__), source / 'report.json', auditpath)}
    totals, available_reasons = collections.Counter(), collections.Counter()
    records = []
    for entry in load(BASE / 'train/report.json')['cases']:
        stem = Path(entry['image']).stem
        prepared_path = PREP / 'train' / (stem + '_seeds.json')
        saved_path = source / 'train' / (stem + '_predictions.json')
        for p in (prepared_path, saved_path):
            pins[str(p)] = sha(p)
        case, saved = load(prepared_path), load(saved_path)
        targets = read_targets('train', entry['image'], [2736, 3648], entry['label_sha256'], pins)
        old_hit = matches(case['current']['all_predictions'], targets)[0]
        for ti, target in enumerate(targets):
            if ti in old_hit:
                continue
            before = vote_map(case['models'], case['source_sha256'], target)
            after = vote_map(case['models'] + saved['extra_models'], case['source_sha256'], target)
            missing = {k for k, v in before.items() if v < .5}
            covered, margins = set(), set()
            best_fraction = 0.
            for evidence in saved['ROI_evidence']:
                for window in evidence['windows']:
                    fraction = target_fraction(target['box'], window)
                    best_fraction = max(best_fraction, fraction)
                    if fraction >= .85:
                        covered.add(evidence['missing_weight_sha256'])
                    l, t, r, b = target['box']
                    a, c, d, e = window
                    if a+16 < l < r < d-16 and c+16 < t < b < e-16:
                        margins.add(evidence['missing_weight_sha256'])
            before_count = sum(v >= .5 for v in before.values())
            after_count = sum(v >= .5 for v in after.values())
            enough_views = missing <= covered
            gain = ti in matches(saved['trial']['all_predictions'], targets)[0]
            if not case['remaining_budget']:
                reason = 'budget_full'
            elif not covered:
                reason = 'no_acquired_view_covers_target'
            elif not enough_views:
                reason = 'not_all_missing_checkpoints_queried_over_target'
            elif after_count < 3:
                reason = 'missing_checkpoints_still_fail_after_covering_ROI'
            elif gain:
                reason = 'gained'
            else:
                reason = 'three_raw_votes_but_no_selected_gain'
            totals[reason] += 1
            if case['remaining_budget']:
                available_reasons[reason] += 1
            records.append(dict(image=entry['image'], target_index=ti, remaining_budget=case['remaining_budget'],
                 before_best_IoU=before, after_best_IoU=after, before_vote_count=before_count,
                 after_vote_count=after_count, missing_checkpoints=sorted(missing),
                 covered_checkpoint_views=sorted(covered), strict_interior_checkpoint_views=sorted(margins),
                 maximum_target_area_covered=best_fraction, diagnostic_reason=reason, gained=gain))
    assert len(records) == 46 and sum(available_reasons.values()) == 16
    assert all(sha(Path(p)) == v for p, v in pins.items())
    out.mkdir()
    result = dict(status='complete', source_report_sha256=sha(source/'report.json'),
                  ALL_TRAIN_research_prefix_misses=46, available_budget_misses=16,
                  reasons=dict(totals), available_budget_reasons=dict(available_reasons),
                  votes_increased=sum(r['after_vote_count'] > r['before_vote_count'] for r in records),
                  cases=records, pins=pins, no_GT_used_by_inference=True,
                  coverage_not_nested_with_seed_IoU=True, no_validation_read=True,
                  no_model_or_policy_changes=True, field_accuracy=False)
    save(out/'report.json', result)
    print({k: v for k, v in result.items() if k not in ('cases', 'pins')}, flush=True)


if __name__ == '__main__':
    main()
