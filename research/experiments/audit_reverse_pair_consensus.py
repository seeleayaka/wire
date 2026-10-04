"""Cached fold/full score and independent geometry-rank replay for all192."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
from prepare_paired_port_semantics import ROOT, REPO, load, save, sha
from current_port_baseline_audit import BASE, read_targets
from inspection_agent.paired_native_pose import native_pose_runtime_fingerprint
from audit_fine_consensus_rank import independent_selection
from audit_port_multiscale_acceptance import metric, matches
SOURCE = ROOT / 'artifacts/reverse_pair_consensus_20261004'
REVERSE = ROOT / 'artifacts/reverse_pair_semantics_20261004'
TRAINING = ROOT / 'artifacts/fine_pose_training_20261004/training'
OUT = ROOT / 'artifacts/reverse_pair_consensus_replay_20261004'


def main():
    import torch
    torch.set_num_threads(2)
    if OUT.exists():
        raise FileExistsError('Preserve reverse geometric replay')
    report = load(SOURCE / 'report.json')
    digest = sha(SOURCE / 'report.json')
    assert all(sha(Path(p)) == v for p, v in report['pins'].items())
    assert native_pose_runtime_fingerprint(REPO) == report['runtime']
    raw = torch.load(TRAINING / 'raw_features.pt', map_location='cpu', weights_only=True)['features']
    records = load(TRAINING / 'raw_samples.json')
    groups = sorted(load(ROOT / 'artifacts/port_training_multiscale_20261002/protocol.json')['train_sources'])
    foldmap = {n: i % 3 for i, n in enumerate(groups)}
    scores, digests, states = {}, {}, {}
    for stage in report['stages']:
        if stage == 'source_train_oof':
            score = torch.zeros((969, 3))
            rawfolds = torch.tensor([r['fold'] for r in records])
            dig = {}
            for fold in range(3):
                path = REVERSE / 'heads_oof' / ('fold'+str(fold)+'.pt')
                state = torch.load(path, map_location='cpu', weights_only=True)
                mask = rawfolds == fold
                score[mask] = (raw[mask] @ state['weight'].T + state['bias']).softmax(1)
                dig[fold] = sha(path)
        else:
            path = SOURCE / 'full/last_head.pt'
            state = torch.load(path, map_location='cpu', weights_only=True)['state_dict']
            score = (raw @ state['weight'].T + state['bias']).softmax(1)
            dig = {i: sha(path) for i in range(3)}
            states[stage] = state
        scores[stage], digests[stage] = score, dig
    pins, counts, summaries = {}, {}, {}
    for stage, probabilities in scores.items():
        rows, offset = [], 0
        for entry in load(BASE / 'train/report.json')['cases']:
            name = entry['image']
            path = SOURCE / stage / (Path(name).stem+'_predictions.json')
            pins[str(path)] = sha(path)
            case = load(path)
            n = len(case['proposals'])
            assert len(records[offset:offset+n]) == n
            assert all(r['image'] == name and r['box'] == p['box_xyxy'] for r, p in zip(records[offset:offset+n], case['proposals']))
            expected = probabilities[offset:offset+n]
            if stage == 'full_train':
                # Same per-image batch shape and fused GEMM as the original
                # driver, not a larger batch with separate bias addition.
                state = states[stage]
                expected = torch.nn.functional.linear(raw[offset:offset+n], state['weight'], state['bias']).softmax(1)
            torch.testing.assert_close(torch.tensor(case['probabilities']).reshape(n, 3), expected, atol=1e-7, rtol=1e-6)
            offset += n
            assert case['head_sha256'] == digests[stage][foldmap[name]]
            assert independent_selection(case['current'], case['proposals'], case['probabilities'], case['head_sha256']) == case['trial']
            gt = read_targets('train', name, [2736, 3648], entry['label_sha256'], pins)
            a, b = matches(case['current']['all_predictions'], gt)[0], matches(case['trial']['all_predictions'], gt)[0]
            rows.append(dict(image=name, current=metric(case['current']['all_predictions'], gt),
                trial=metric(case['trial']['all_predictions'], gt), gained=sorted(b-a), lost=sorted(a-b)))
        assert offset == 969 and rows == load(SOURCE / stage / 'report.json')['cases']
        totals = {v: {k: sum(r[v][k] for r in rows) for k in ('tp', 'unmatched', 'fn', 'predictions', 'targets')} for v in ('current', 'trial')}
        assert totals == report['stages'][stage]['summary']
        counts[stage], summaries[stage] = len(rows), totals
    assert all(sha(Path(p)) == v for p, v in report['pins'].items()) and sha(SOURCE / 'report.json') == digest
    OUT.mkdir()
    result = dict(status='pass', counts=counts, summaries=summaries, candidates_per_stage=969,
                  independent_head_score_rank_duplicate_GT_replay=True, source_report_sha256=digest,
                  no_deployment=True, field_accuracy=False, pins=pins)
    save(OUT / 'report.json', result)
    print({k: v for k, v in result.items() if k != 'pins'}, flush=True)


if __name__ == '__main__':
    main()
