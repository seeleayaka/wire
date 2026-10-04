"""Finite cached TRAIN-only reverse-pair augmentation, unchanged inference."""
import os
import sys
import time
from pathlib import Path
sys.dont_write_bytecode = True
from prepare_paired_port_semantics import ROOT, REPO, load, save, sha
from current_port_baseline_audit import BASE, read_targets
from inspection_agent.paired_native_pose import native_pose_runtime_fingerprint
from inspection_agent.paired_native_pose_features import select
from audit_port_multiscale_acceptance import metric, matches
from reverse_pair_semantics import augment
from port_semantic_verifier import fit_head

OLD = ROOT / 'artifacts/fine_pose_training_20261004'
TRAINING = OLD / 'training'
SOURCE = ROOT / 'artifacts/fine_native_consensus_20261004/train'
OUT = ROOT / 'artifacts/reverse_pair_semantics_20261004'
PLAN = ROOT / 'artifacts/reverse_pair_semantics_preregistration_20261004/PLAN.md'


def main():
    import torch
    import psutil
    torch.set_num_threads(2)
    if OUT.exists():
        raise FileExistsError('Preserve finite reverse pair trial')
    assert psutil.virtual_memory().available > 6*2**30
    previous = load(OLD / 'report.json')
    auditpath = ROOT / 'artifacts/fine_pose_training_audit_20261004/report.json'
    audit = load(auditpath)
    assert previous['status'] == 'rejected' and audit['status'] == 'complete'
    assert audit['source_report_sha256'] == sha(OLD / 'report.json')
    assert all(sha(Path(p)) == v for p, v in previous['pins'].items())
    runtime = native_pose_runtime_fingerprint(REPO)
    assert runtime == previous['runtime']
    groupfile = ROOT / 'artifacts/port_training_multiscale_20261002/protocol.json'
    groups = load(groupfile)
    names = sorted(groups['train_sources'])
    assert len(names) == 192 and set(names).isdisjoint(groups['inner_val_sources'])
    source_folds = {n: i % 3 for i, n in enumerate(names)}
    paths = [Path(__file__), PLAN, groupfile, OLD / 'report.json', auditpath,
             Path(__file__).with_name('reverse_pair_semantics.py'),
             Path(__file__).with_name('port_semantic_verifier.py'),
             REPO / 'inspection_agent/paired_native_pose_features.py',
             REPO / 'inspection_agent/paired_port_features.py',
             *[TRAINING / n for n in ('features.pt', 'samples.json', 'raw_features.pt', 'raw_samples.json')]]
    pins = {str(p): sha(p) for p in paths}
    data = torch.load(TRAINING / 'features.pt', map_location='cpu', weights_only=True)
    raw = torch.load(TRAINING / 'raw_features.pt', map_location='cpu', weights_only=True)['features']
    samples, records = load(TRAINING / 'samples.json'), load(TRAINING / 'raw_samples.json')
    features, labels, folds = data['features'], data['labels'], data['folds']
    assert features.shape == (9808, 6144) and raw.shape == (969, 6144)
    assert all(row['fold'] == source_folds[row['image']] == int(folds[i]) and row['label'] == int(labels[i]) for i, row in enumerate(samples))
    assert len(samples) == 9808 and len(records) == 969
    assert all(row['fold'] == source_folds[row['image']] for row in records)
    augmented, targets, sourcegroups, reversed_indices = augment(features, labels, folds)
    OUT.mkdir()
    start = time.monotonic()
    stages = {}
    def progress(**kw):
        save(OUT / 'progress.json', dict(status='running', pid=os.getpid(), seconds=round(time.monotonic()-start, 2), **kw))
    save(OUT / 'protocol.json', dict(pins=pins, runtime=runtime, train_sources=names, source_folds=source_folds,
         original_samples=9808, reverse_positive_samples=len(reversed_indices), dimensions=6144,
         steps=400, seed=0, source_classifier_only_OOF=True, detector_and_prefix_not_OOF=True,
         no_validation_read=True, no_deployment=True, field_accuracy=False))
    save(OUT / 'augmentation.json', dict(original_indices=reversed_indices.tolist(),
         original_label_counts=torch.bincount(labels, minlength=3).tolist(),
         augmented_label_counts=torch.bincount(targets, minlength=3).tolist(),
         reverse_twins_all_source_folds=sourcegroups[len(features):].tolist()))
    def evaluate(stage, scores, digests):
        folder = OUT / stage
        folder.mkdir()
        rows, offset = [], 0
        for entry in load(BASE / 'train/report.json')['cases']:
            name = entry['image']
            path = SOURCE / (Path(name).stem + '_predictions.json')
            pins[str(path)] = sha(path)
            case = load(path)
            n = len(case['proposals'])
            local = records[offset:offset+n]
            assert len(local) == n and all(r['image'] == name and r['box'] == p['box_xyxy'] for r, p in zip(local, case['proposals']))
            probabilities = scores[offset:offset+n].tolist()
            offset += n
            digest = digests[source_folds[name]]
            trial = select(case['current'], case['proposals'], probabilities, digest)
            gt = read_targets('train', name, [2736, 3648], entry['label_sha256'], pins)
            a, b = matches(case['current']['all_predictions'], gt)[0], matches(trial['all_predictions'], gt)[0]
            rows.append(dict(image=name, current=metric(case['current']['all_predictions'], gt),
                             trial=metric(trial['all_predictions'], gt), gained=sorted(b-a), lost=sorted(a-b)))
            save(folder / path.name, dict(image=name, current=case['current'], trial=trial, proposals=case['proposals'],
                 probabilities=probabilities, head_sha256=digest))
        assert offset == len(records) == 969
        totals = {v: {k: sum(r[v][k] for r in rows) for k in ('tp', 'unmatched', 'fn', 'predictions', 'targets')} for v in ('current', 'trial')}
        assert totals['current']['tp'] == 295 and totals['current']['unmatched'] == 4
        normal = sum(r['trial']['predictions'] for r in rows if r['image'].startswith('normal_'))
        qualifies = totals['trial']['tp'] > 295 and totals['trial']['unmatched'] <= 4 and normal == 0 and not any(r['lost'] for r in rows)
        summary = dict(qualifies=qualifies, summary=totals, normal_cues=normal)
        stages[stage] = summary
        save(folder / 'report.json', dict(status='complete', **summary, cases=rows))
        print(dict(stage=stage, **summary), flush=True)
        return qualifies
    def finish(failed):
        assert all(sha(Path(p)) == v for p, v in pins.items()) and native_pose_runtime_fingerprint(REPO) == runtime
        result = dict(status='rejected' if failed else 'source_pass_requires_fresh_holdouts_and_actual_gates',
                      failed_stage=failed, stages=stages, pins=pins, runtime=runtime,
                      classifier_only_source_OOF=True, no_validation_read=True, no_deployment=True,
                      field_accuracy=False, seconds=round(time.monotonic()-start, 2))
        save(OUT / 'report.json', result)
        save(OUT / 'progress.json', dict(status=result['status'], seconds=result['seconds']))
    try:
        heads = OUT / 'heads_oof'
        heads.mkdir()
        rawfolds = torch.tensor([r['fold'] for r in records])
        scores = torch.zeros((969, 3))
        digests = {}
        for fold in range(3):
            progress(phase='fixed400_reverse_pair_source_OOF', fold=fold)
            mask = sourcegroups != fold
            for j, original in enumerate(reversed_indices.tolist()):
                assert bool(mask[original]) == bool(mask[9808+j])
            head = fit_head(augmented[mask], targets[mask])
            path = heads / ('fold'+str(fold)+'.pt')
            torch.save(head.state_dict(), path)
            digests[fold] = sha(path)
            pins[str(path)] = digests[fold]
            with torch.inference_mode():
                scores[rawfolds == fold] = head(raw[rawfolds == fold]).softmax(1)
        torch.save(dict(probabilities=scores), heads / 'raw_probabilities.pt')
        if not evaluate('source_train_oof', scores, digests):
            finish('source_train_oof')
            return
        progress(phase='fixed400_reverse_pair_full_head')
        head = fit_head(augmented, targets)
        full = OUT / 'full'
        full.mkdir()
        path = full / 'last_head.pt'
        torch.save(dict(state_dict=head.state_dict(), input_dimensions=6144,
                   classes=['other', 'unplugged_plug', 'unplugged_jack'], encoder_sha256=runtime['encoder']), path)
        pins[str(path)] = sha(path)
        with torch.inference_mode():
            scores = head(raw).softmax(1)
        if not evaluate('full_train', scores, {i: pins[str(path)] for i in range(3)}):
            finish('full_train')
            return
        finish(None)
    except BaseException as error:
        save(OUT / 'progress.json', dict(status='failed', error=type(error).__name__+': '+str(error)))
        raise


if __name__ == '__main__':
    main()
