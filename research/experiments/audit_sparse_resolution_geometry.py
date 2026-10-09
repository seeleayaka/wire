"""Replay ROI evidence, distinct voters, ranking and metrics; no model inference.

Saved semantic probabilities are replayed, NOT independently re-extracted.
Final inference accuracy remains subject to fresh holdouts and actual acceptance.
"""
import copy
import statistics
import sys
from pathlib import Path
sys.dont_write_bytecode = True
from prepare_paired_port_semantics import ROOT, REPO, load, save, sha
from current_port_baseline_audit import BASE, read_targets
from inspection_agent.paired_native_pose import HEAD_SHA, native_pose_runtime_fingerprint
from audit_fine_consensus_rank import duplicate, iou, independent_selection
from audit_port_multiscale_acceptance import metric, matches

SOURCE = ROOT / 'artifacts/sparse_nomination_resolution_20261004'
PREP = ROOT / 'artifacts/sparse_nomination_20261004'
OUT = ROOT / 'artifacts/sparse_nomination_resolution_replay_20261004'


def prescreen(case, probabilities):
    assert len(case['seeds']) == len(probabilities)
    best = {}
    for seed, prob in zip(case['seeds'], probabilities):
        assert len(prob) == 3 and all(0 <= p <= 1 for p in prob)
        assert abs(sum(prob) - 1) < 1e-5
        cls = max(range(3), key=lambda i: prob[i])
        if cls != seed['class_id'] + 1 or prob[cls] < .98:
            continue
        assert len(set(seed['semantic_model_vote_sha256'])) == 2
        quality = statistics.median(seed['localization_voter_best_IoU'].values())
        rank = (-quality, -prob[cls], *seed['box_xyxy'], seed['class_id'])
        parent = seed['pose_parent_seed_id']
        if parent not in best or rank < best[parent][0]:
            best[parent] = (rank, seed, prob)
    chosen = []
    for rank, seed, prob in sorted(best.values(), key=lambda r: r[0]):
        if len(chosen) >= case['remaining_budget']:
            break
        if duplicate(seed['box_xyxy'], chosen):
            continue
        row = copy.deepcopy(seed)
        row['prescreen_probability'] = prob[seed['class_id'] + 1]
        chosen.append(row)
    return chosen


def verify_case(case, saved):
    assert saved['image'] == case['image']
    assert saved['source_sha256'] == case['source_sha256']
    assert saved['current'] == case['current'] and saved['seeds'] == case['seeds']
    assert saved['alignment'] == case['alignment'] and saved['head_sha256'] == HEAD_SHA
    chosen = prescreen(case, saved['seed_probabilities'])
    assert chosen == saved['chosen_seeds']
    assert 3*len(chosen) == len(saved['ROI_evidence']) == len(saved['extra_models'])
    assert saved['acquisition_all_three_checkpoints']
    controlpath=ROOT/'artifacts/sparse_nomination_20261004/train'/(Path(case['image']).stem+'_nominations.json')
    control=load(controlpath)
    assert control['chosen_seeds']==chosen and control['source_sha256']==case['source_sha256']
    shapes = {tuple(m['predictions']['source_shape']) for m in case['models']}
    assert len(shapes) == 1
    h, w = next(iter(shapes))
    weights = {m['weight_sha256'] for m in case['models']}
    assert len(weights) == 3
    requests=[(seed,weight) for seed in chosen for weight in sorted(weights)]
    for (seed,weight), evidence, model in zip(requests, saved['ROI_evidence'], saved['extra_models']):
        missing = weights - set(seed['semantic_model_vote_sha256'])
        assert len(missing) in (1,2)
        assert evidence['seed'] == seed
        assert evidence['requested_weight_sha256'] == model['weight_sha256'] == weight
        assert evidence['reused_control_views'] is False
        assert model['source_sha256'] == case['source_sha256'] and model['image'] == case['image']
        assert model['predictions']['source_shape'] == [h, w]
        l, t, r, b = seed['box_xyxy']
        expected = []
        for delta in (-120, 120):
            x = max(0, min(w - 960, round((l + r) / 2 + delta - 480)))
            y = max(0, min(h - 960, round((t + b) / 2 + delta - 480)))
            expected.append([x, y, x + 960, y + 960])
        assert evidence['windows'] == expected and len(evidence['views']) == 2
        assert [row for view in evidence['views'] for row in view] == model['predictions']['merged_predictions']
        for window, view in zip(expected, evidence['views']):
            x, y, rr, bb = window
            for row in view:
                a, c, d, e = row['box_xyxy']
                assert x <= a < d <= rr and y <= c < e <= bb
                assert not ((x > 0 and a-x <= 16) or (y > 0 and c-y <= 16)
                            or (rr < w and d-x >= 944) or (bb < h and e-y >= 944))
                assert row['class_id'] in (0, 1) and 0 <= row['confidence'] <= 1
    all_models = case['models'] + saved['extra_models']
    for candidate in saved['proposals']:
        voters = {}
        for model in all_models:
            for row in model['predictions']['merged_predictions']:
                l, t, r, b = row['box_xyxy']
                if row['class_id'] != candidate['class_id'] or row['confidence'] <= .05:
                    continue
                if not (16 <= l < r <= w-16 and 16 <= t < b <= h-16):
                    continue
                overlap = iou(candidate['box_xyxy'], row['box_xyxy'])
                if overlap >= .5:
                    key = model['weight_sha256']
                    voters[key] = max(overlap, voters.get(key, 0.))
        assert len(voters) == 3 and sorted(voters) == candidate['semantic_model_vote_sha256']
        assert voters == candidate['localization_voter_best_IoU']
    assert len(saved['proposals']) == len(saved['probabilities'])
    trial = independent_selection(case['current'], saved['proposals'], saved['probabilities'], HEAD_SHA)
    assert trial == saved['trial']
    assert trial['all_predictions'][:len(case['current']['all_predictions'])] == case['current']['all_predictions']
    assert len(trial['primary']) <= 5 and len(trial['all_predictions']) <= len(trial['primary']) + 5
    return len(chosen), len(saved['proposals'])


def main():
    if OUT.exists():
        raise FileExistsError('Preserve complete ROI replay')
    report = load(SOURCE / 'report.json')
    protocol = load(SOURCE / 'protocol.json')
    prepared = load(PREP / 'report.json')
    digest = sha(SOURCE / 'report.json')
    assert report['status'] in ('rejected', 'source_pass_requires_combined_fresh_holdouts_and_actual_gates')
    assert all(sha(Path(p)) == value for p, value in report['pins'].items())
    assert native_pose_runtime_fingerprint(REPO) == report['runtime'] == prepared['runtime']
    entries = load(BASE / 'train/report.json')['cases']
    assert len(entries) == len(prepared['cases']) == 192
    rows = []
    pins = {}
    roi_seeds = 0
    candidates = 0
    for entry, record in zip(entries, prepared['cases']):
        assert entry['image'] == record['image']
        path = Path(record['path'])
        assert sha(path) == record['sha256']
        case = load(path)
        output = SOURCE / 'train' / (Path(entry['image']).stem + '_predictions.json')
        pins[str(output)] = sha(output)
        saved = load(output)
        n, c = verify_case(case, saved)
        roi_seeds += n
        candidates += c
        targets = read_targets('train', entry['image'], [2736, 3648], entry['label_sha256'], pins)
        a = matches(saved['current']['all_predictions'], targets)[0]
        b = matches(saved['trial']['all_predictions'], targets)[0]
        rows.append(dict(image=entry['image'], current=metric(saved['current']['all_predictions'], targets),
                         trial=metric(saved['trial']['all_predictions'], targets), gained=sorted(b-a), lost=sorted(a-b), ROI_seeds=n))
    train = load(SOURCE / 'train/report.json')
    assert rows == train['cases']
    totals = {v: {k: sum(r[v][k] for r in rows) for k in ('tp', 'unmatched', 'fn', 'predictions', 'targets')}
              for v in ('current', 'trial')}
    normal = sum(r['trial']['predictions'] for r in rows if r['image'].startswith('normal_'))
    qualifies = totals['trial']['tp'] > 298 and totals['trial']['unmatched'] <= 4 and not any(r['lost'] for r in rows) and normal == 0
    summary = report['summary']
    assert totals == summary['summary'] and qualifies == summary['qualifies'] and normal == summary['normal_cues']
    assert roi_seeds == summary['ROI_seeds'] and candidates == summary['final_proposals']
    assert sha(SOURCE / 'report.json') == digest
    assert all(sha(Path(p)) == value for p, value in pins.items())
    assert all(sha(Path(p)) == value for p, value in report['pins'].items())
    OUT.mkdir()
    result = dict(status='pass', sources=192, source_report_sha256=digest, summary=summary,
                  ROI_seeds=roi_seeds, candidates=candidates, independent_seed_window_vote_rank_GT_replay=True,
                  semantic_probabilities_recomputed=False, detector_predictions_reinferred=False,
                  no_deployment=True, field_accuracy=False, pins=pins, auditor_sha256=sha(Path(__file__)))
    save(OUT / 'report.json', result)
    print({k: v for k, v in result.items() if k != 'pins'}, flush=True)


if __name__ == '__main__':
    main()
