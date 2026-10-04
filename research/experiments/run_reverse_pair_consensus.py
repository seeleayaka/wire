"""Fixed geometric filter on audited reverse-pair OOF, then at most one head."""
import os
import sys
import time
from pathlib import Path
sys.dont_write_bytecode = True
from prepare_paired_port_semantics import ROOT, REPO, load, save, sha
from current_port_baseline_audit import BASE, read_targets
from inspection_agent.paired_native_pose import native_pose_runtime_fingerprint
from audit_port_multiscale_acceptance import metric, matches
from consensus_rank import select
from reverse_pair_semantics import augment
from port_semantic_verifier import fit_head

REVERSE = ROOT / 'artifacts/reverse_pair_semantics_20261004'
GEOMETRY = ROOT / 'artifacts/fine_consensus_rank_20261004'
TRAINING = ROOT / 'artifacts/fine_pose_training_20261004/training'
OUT = ROOT / 'artifacts/reverse_pair_consensus_20261004'
PLAN = ROOT / 'artifacts/reverse_pair_consensus_preregistration_20261004/PLAN.md'


def main():
    import torch
    torch.set_num_threads(2)
    if OUT.exists():
        raise FileExistsError('Preserve fixed reverse geometry cascade')
    previous = load(REVERSE / 'report.json')
    auditpath = ROOT / 'artifacts/reverse_pair_semantics_replay_20261004/report.json'
    assert load(auditpath)['status'] == 'pass' and load(auditpath)['source_report_sha256'] == sha(REVERSE / 'report.json')
    frozen = native_pose_runtime_fingerprint(REPO)
    assert previous['runtime'] == frozen
    assert all(sha(Path(p)) == v for p, v in previous['pins'].items())
    geometry = load(GEOMETRY / 'report.json')
    assert geometry['runtime'] == frozen and all(sha(Path(p)) == v for p, v in geometry['pins'].items())
    raw = torch.load(TRAINING / 'raw_features.pt', map_location='cpu', weights_only=True)['features']
    records = load(TRAINING / 'raw_samples.json')
    groupfile = ROOT / 'artifacts/port_training_multiscale_20261002/protocol.json'
    foldmap = {n: i % 3 for i, n in enumerate(sorted(load(groupfile)['train_sources']))}
    pins = {str(p): sha(p) for p in (Path(__file__), PLAN, REVERSE / 'report.json', auditpath,
        GEOMETRY / 'report.json', groupfile, TRAINING / 'features.pt', TRAINING / 'raw_features.pt',
        TRAINING / 'raw_samples.json', Path(__file__).with_name('consensus_rank.py'),
        Path(__file__).with_name('novel_box_geometry.py'), Path(__file__).with_name('reverse_pair_semantics.py'),
        Path(__file__).with_name('port_semantic_verifier.py'))}
    OUT.mkdir()
    start, stages = time.monotonic(), {}
    def progress(**kw):
        save(OUT / 'progress.json', dict(status='running', pid=os.getpid(), seconds=round(time.monotonic()-start, 2), **kw))
    def evaluate(stage, head=None, digest=None):
        folder = OUT / stage
        folder.mkdir()
        rows, offset = [], 0
        for entry in load(BASE / 'train/report.json')['cases']:
            name = entry['image']
            path = REVERSE / 'source_train_oof' / (Path(name).stem+'_predictions.json')
            geop = GEOMETRY / 'train' / path.name
            pins[str(path)], pins[str(geop)] = sha(path), sha(geop)
            case, geo = load(path), load(geop)
            assert case['current'] == geo['current']
            cleaned = [dict(p) for p in geo['proposals']]
            for p in cleaned:
                p.pop('localization_voter_best_IoU')
            assert cleaned == case['proposals']
            n = len(cleaned)
            local = records[offset:offset+n]
            assert all(r['image'] == name and r['box'] == p['box_xyxy'] for r, p in zip(local, cleaned)) and len(local) == n
            if head is None:
                probabilities, hsha = case['probabilities'], case['head_sha256']
            else:
                with torch.inference_mode():
                    probabilities = head(raw[offset:offset+n]).softmax(1).tolist()
                hsha = digest
            offset += n
            trial = select(case['current'], geo['proposals'], probabilities, hsha)
            gt = read_targets('train', name, [2736, 3648], entry['label_sha256'], pins)
            a, b = matches(case['current']['all_predictions'], gt)[0], matches(trial['all_predictions'], gt)[0]
            rows.append(dict(image=name, current=metric(case['current']['all_predictions'], gt),
                             trial=metric(trial['all_predictions'], gt), gained=sorted(b-a), lost=sorted(a-b)))
            save(folder / path.name, dict(image=name, current=case['current'], trial=trial,
                 proposals=geo['proposals'], probabilities=probabilities, head_sha256=hsha))
        assert offset == 969 and len(rows) == 192
        totals = {v: {k: sum(r[v][k] for r in rows) for k in ('tp', 'unmatched', 'fn', 'predictions', 'targets')} for v in ('current', 'trial')}
        normal = sum(r['trial']['predictions'] for r in rows if r['image'].startswith('normal_'))
        qualifies = totals['trial']['tp'] > 295 and totals['trial']['unmatched'] <= 4 and normal == 0 and not any(r['lost'] for r in rows)
        summary = dict(qualifies=qualifies, summary=totals, normal_cues=normal)
        stages[stage] = summary
        save(folder / 'report.json', dict(status='complete', **summary, cases=rows))
        print(dict(stage=stage, **summary), flush=True)
        return qualifies
    def finish(failed):
        assert all(sha(Path(p)) == v for p, v in pins.items()) and native_pose_runtime_fingerprint(REPO) == frozen
        status = 'rejected' if failed else 'source_pass_requires_fresh_holdouts_and_actual_gates'
        save(OUT / 'report.json', dict(status=status, failed_stage=failed, stages=stages, pins=pins,
             runtime=frozen, no_validation_fitting=True, no_deployment=True, field_accuracy=False,
             seconds=round(time.monotonic()-start, 2)))
        save(OUT / 'progress.json', dict(status=status, seconds=round(time.monotonic()-start, 2)))
    progress(phase='cached_reverse_OOF_fixed_geometry')
    if not evaluate('source_train_oof'):
        finish('source_train_oof')
        return
    progress(phase='single_fixed400_augmented_full_head')
    data = torch.load(TRAINING / 'features.pt', map_location='cpu', weights_only=True)
    x, y, folds, indices = augment(data['features'], data['labels'], data['folds'])
    head = fit_head(x, y)
    full = OUT / 'full'
    full.mkdir()
    path = full / 'last_head.pt'
    torch.save(dict(state_dict=head.state_dict(), input_dimensions=6144,
        classes=['other', 'unplugged_plug', 'unplugged_jack'], encoder_sha256=frozen['encoder']), path)
    pins[str(path)] = sha(path)
    if not evaluate('full_train', head, pins[str(path)]):
        finish('full_train')
        return
    finish(None)


if __name__ == '__main__':
    main()
