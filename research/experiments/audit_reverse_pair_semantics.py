"""Reconstruct reverse-pair augmentation, source folds and saved head scores."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
from prepare_paired_port_semantics import ROOT, REPO, load, save, sha
from current_port_baseline_audit import BASE, read_targets
from audit_port_multiscale_acceptance import metric, matches
from inspection_agent.paired_native_pose import native_pose_runtime_fingerprint
from inspection_agent.paired_port_features import select as append_original
from reverse_pair_semantics import augment

SOURCE = ROOT / 'artifacts/reverse_pair_semantics_20261004'
TRAINING = ROOT / 'artifacts/fine_pose_training_20261004/training'
OUT = ROOT / 'artifacts/reverse_pair_semantics_replay_20261004'


def native_selection(current, proposals, probabilities, head):
    chosen = {}
    for row, prob in zip(proposals, probabilities):
        cls = max(range(3), key=lambda i: prob[i])
        if cls != row['class_id']+1 or prob[cls] < .98 or len(set(row['semantic_model_vote_sha256'])) < 3:
            continue
        rank = (-prob[cls], *row['box_xyxy'], row['class_id'])
        parent = row['pose_parent_seed_id']
        if parent not in chosen or rank < chosen[parent][0]:
            chosen[parent] = (rank, row, prob)
    return append_original(current, [r[1] for r in chosen.values()], [r[2] for r in chosen.values()], head)


def main():
    import torch
    torch.set_num_threads(2)
    if OUT.exists():
        raise FileExistsError('Preserve reverse pair audit')
    report = load(SOURCE / 'report.json')
    digest = sha(SOURCE / 'report.json')
    assert all(sha(Path(p)) == v for p, v in report['pins'].items())
    assert report['runtime'] == native_pose_runtime_fingerprint(REPO)
    data = torch.load(TRAINING / 'features.pt', map_location='cpu', weights_only=True)
    x, y, f = data['features'], data['labels'], data['folds']
    metadata = load(TRAINING / 'samples.json')
    samples = load(TRAINING / 'raw_samples.json')
    raw = torch.load(TRAINING / 'raw_features.pt', map_location='cpu', weights_only=True)['features']
    positive = [i for i, r in enumerate(metadata) if r['label'] > 0]
    record = load(SOURCE / 'augmentation.json')
    assert record['original_indices'] == positive
    ii = torch.tensor(positive)
    expected = torch.cat((x[ii, 1536:3072], x[ii, :1536], x[ii, 3072:]), dim=1)
    xx, yy, ff, indices = augment(x, y, f)
    assert torch.equal(xx[:len(x)], x) and torch.equal(xx[len(x):], expected)
    assert torch.equal(yy[:len(y)], y) and (yy[len(y):] == 0).all()
    assert torch.equal(ff[len(f):], f[ii]) and indices.tolist() == positive
    assert record['augmented_label_counts'] == torch.bincount(yy, minlength=3).tolist()
    assert record['reverse_twins_all_source_folds'] == f[ii].tolist()
    names = sorted(load(ROOT / 'artifacts/port_training_multiscale_20261002/protocol.json')['train_sources'])
    foldmap = {n: i % 3 for i, n in enumerate(names)}
    assert all(r['fold'] == foldmap[r['image']] == int(f[i]) for i, r in enumerate(metadata))
    rawfolds = torch.tensor([r['fold'] for r in samples])
    oof = torch.zeros((969, 3))
    heads = {}
    for fold in range(3):
        state = torch.load(SOURCE / 'heads_oof' / ('fold'+str(fold)+'.pt'), map_location='cpu', weights_only=True)
        heads[fold] = sha(SOURCE / 'heads_oof' / ('fold'+str(fold)+'.pt'))
        mask = rawfolds == fold
        oof[mask] = torch.softmax(raw[mask] @ state['weight'].T + state['bias'], dim=1)
        held = {r['image'] for r in metadata if r['fold'] == fold}
        fit = {r['image'] for r in metadata if r['fold'] != fold}
        assert held.isdisjoint(fit)
        assert all((int(f[i]) != fold) == (int(ff[len(f)+j]) != fold) for j, i in enumerate(positive))
    saved_oof = torch.load(SOURCE / 'heads_oof/raw_probabilities.pt', map_location='cpu', weights_only=True)['probabilities']
    torch.testing.assert_close(saved_oof, oof, atol=1e-7, rtol=1e-6)
    rows, pins, offset = [], {}, 0
    entries = load(BASE / 'train/report.json')['cases']
    savedrows = load(SOURCE / 'source_train_oof/report.json')['cases']
    for entry, row in zip(entries, savedrows):
        name = entry['image']
        path = SOURCE / 'source_train_oof' / (Path(name).stem+'_predictions.json')
        pins[str(path)] = sha(path)
        case = load(path)
        n = len(case['proposals'])
        assert name == row['image'] == case['image']
        assert case['head_sha256'] == heads[foldmap[name]]
        assert [r['box'] for r in samples[offset:offset+n]] == [p['box_xyxy'] for p in case['proposals']]
        assert all(r['image'] == name for r in samples[offset:offset+n])
        torch.testing.assert_close(torch.tensor(case['probabilities']).reshape(n, 3), oof[offset:offset+n], atol=1e-7, rtol=1e-6)
        offset += n
        assert native_selection(case['current'], case['proposals'], case['probabilities'], case['head_sha256']) == case['trial']
        targets = read_targets('train', name, [2736, 3648], entry['label_sha256'], pins)
        for v in ('current', 'trial'):
            assert metric(case[v]['all_predictions'], targets) == row[v]
        a, b = matches(case['current']['all_predictions'], targets)[0], matches(case['trial']['all_predictions'], targets)[0]
        assert row['gained'] == sorted(b-a) and row['lost'] == sorted(a-b)
        rows.append(row)
    assert offset == 969 and len(rows) == 192
    totals = {v: {k: sum(r[v][k] for r in rows) for k in ('tp', 'unmatched', 'fn', 'predictions', 'targets')} for v in ('current', 'trial')}
    assert totals == report['stages']['source_train_oof']['summary']
    assert all(sha(Path(p)) == v for p, v in report['pins'].items()) and sha(SOURCE / 'report.json') == digest
    assert native_pose_runtime_fingerprint(REPO) == report['runtime']
    OUT.mkdir()
    result = dict(status='pass', sources=192, candidates=969, reverse_positive_twins=len(positive),
                  independent_augmentation_fold_score_parent_rank_GT_replay=True,
                  original_feature_extraction_not_repeated=True, optimizer_not_independently_retrained=True,
                  source_report_sha256=digest, summary=totals, no_deployment=True, field_accuracy=False, pins=pins)
    save(OUT / 'report.json', result)
    print({k: v for k, v in result.items() if k != 'pins'}, flush=True)


if __name__ == '__main__':
    main()
