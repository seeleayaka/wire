"""Training-only limits of additive review, with oracle geometry for diagnosis."""
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
OUT = ROOT / 'artifacts/additive_capacity_20261004'


def main():
    if OUT.exists():
        raise FileExistsError('Preserve capacity analysis')
    prepared = load(PREP / 'report.json')
    assert native_pose_runtime_fingerprint(REPO) == prepared['runtime']
    assert all(sha(Path(p)) == v for p, v in prepared['pins'].items())
    pins = {str(p): sha(p) for p in (Path(__file__), PREP / 'report.json')}
    records = {r['image']: r for r in prepared['cases']}
    counts = collections.Counter()
    cases = []
    current_tp = ceiling = targets_total = 0
    for entry in load(BASE / 'train/report.json')['cases']:
        record = records[entry['image']]
        path = Path(record['path'])
        assert sha(path) == record['sha256']
        pins[str(path)] = record['sha256']
        case = load(path)
        targets = read_targets('train', entry['image'], [2736, 3648], entry['label_sha256'], pins)
        hit = matches(case['current']['all_predictions'], targets)[0]
        remaining = 5 - (len(case['current']['all_predictions']) - len(case['current']['primary']))
        assert remaining == case['remaining_budget'] and remaining >= 0
        pool = pool_rows(case['models'], case['source_sha256'], (2736, 3648))
        missed = []
        for i, target in enumerate(targets):
            if i in hit:
                continue
            best_by_weight = {}
            for weight, row in pool:
                if row['class_id'] == target['class_id']:
                    value = overlap(row['box_xyxy'], target['box'])
                    best_by_weight[weight] = max(value, best_by_weight.get(weight, 0.))
            votes = sum(v >= .5 for v in best_by_weight.values())
            reason = 'budget_full' if remaining == 0 else 'available_budget_raw_votes_' + str(votes)
            counts[reason] += 1
            missed.append(dict(target_index=i, diagnostic_reason=reason, best_IoU_per_checkpoint=best_by_weight))
        optimistic_additions = min(remaining, len(missed))
        current_tp += len(hit)
        ceiling += len(hit) + optimistic_additions
        targets_total += len(targets)
        cases.append(dict(image=entry['image'], current_tp=len(hit), remaining_budget=remaining,
                          misses=missed, optimistic_extra_matches=optimistic_additions))
    assert len(cases) == 192 and current_tp == 298 and sum(counts.values()) == 46 and targets_total == 344
    assert all(sha(Path(p)) == v for p, v in pins.items())
    OUT.mkdir()
    result = dict(status='complete', cases=cases, counts=dict(counts), current_tp=current_tp,
                  optimistic_budget_only_ceiling=ceiling, targets=targets_total,
                  oracle_upper_bound_not_achieved_accuracy=True,
                  GT_only_for_diagnosis_not_used_in_inference=True,
                  no_validation_read=True, no_model_or_policy_changes=True,
                  runtime=prepared['runtime'], pins=pins)
    save(OUT / 'report.json', result)
    print({k: v for k, v in result.items() if k not in ('cases', 'pins', 'runtime')})


if __name__ == '__main__':
    main()
