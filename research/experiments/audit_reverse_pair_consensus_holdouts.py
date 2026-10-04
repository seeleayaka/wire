"""Held-out score/voter/rank/GT replay from pinned fresh paired vectors."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
from prepare_paired_port_semantics import ROOT, REPO, DATA, load, save, sha
from current_port_baseline_audit import BASE, read_current_case, read_targets
from inspection_agent.paired_native_pose import native_pose_runtime_fingerprint, HEAD_RELATIVE
from audit_fine_consensus_rank import independent_selection, iou
from audit_port_multiscale_acceptance import metric, matches

SOURCE = ROOT / 'artifacts/reverse_pair_consensus_holdouts_20261004'
PREVIOUS = ROOT / 'artifacts/fine_consensus_rank_holdouts_20261004'
PAIRINDEX = ROOT / 'artifacts/paired_support_graph_20261003/source_selections'
OUT = ROOT / 'artifacts/reverse_pair_consensus_holdouts_replay_20261004'


def main():
    import torch
    torch.set_num_threads(2)
    if OUT.exists():
        raise FileExistsError('Preserve combined held-out replay')
    report = load(SOURCE / 'report.json')
    digest = sha(SOURCE / 'report.json')
    assert report['status'] in ('rejected', 'holdout_pass_requires_actual_reference_ROI_and_Qt_SAM')
    assert all(sha(Path(p)) == v for p, v in report['pins'].items())
    assert native_pose_runtime_fingerprint(REPO) == report['runtime']
    headpath = ROOT / 'artifacts/reverse_pair_consensus_20261004/full/last_head.pt'
    state = torch.load(headpath, map_location='cpu', weights_only=True)['state_dict']
    oldstate = torch.load(REPO / HEAD_RELATIVE, map_location='cpu', weights_only=True)['state_dict']
    pins, summaries, counts, total_features = {}, {}, {}, 0
    for stage in report['stages']:
        entries = load(BASE / stage / 'report.json')['cases']
        indexed = {r['image']: r for r in load(PAIRINDEX / stage / 'index.json')['records']}
        rows = []
        for entry in entries:
            name = entry['image']
            path = SOURCE / stage / (Path(name).stem+'_predictions.json')
            pins[str(path)] = sha(path)
            case = load(path)
            source = DATA / 'images' / ('val01' if stage == 'outer' else 'train01') / name
            assert sha(source) == case['source_sha256'] == report['pins'][str(source)]
            assert case['head_sha256'] == sha(headpath)
            old_case = load(PREVIOUS / stage / path.name) if stage == 'inner' else None
            if old_case is not None:
                assert case['current'] == old_case['current'] and case['proposals'] == old_case['proposals']
                assert case['new_views'] == old_case['new_views'] and case['alignment'] == old_case['alignment']
                assert case['detector_reused'] and case['eligible'] == old_case['eligible']
            probabilities = []
            if case['proposals']:
                featurepath = Path(case['feature_file'])
                assert sha(featurepath) == report['pins'][str(featurepath)]
                feature = torch.load(featurepath, map_location='cpu', weights_only=True)
                x = feature['features']
                assert feature['source_sha256'] == case['source_sha256']
                assert feature['boxes'] == [p['box_xyxy'] for p in case['proposals']]
                assert x.shape == (len(case['proposals']), 6144) and torch.isfinite(x).all()
                torch.testing.assert_close(x[:, 3072:4608], (x[:, :1536]-x[:, 1536:3072]).abs(), atol=0, rtol=0)
                torch.testing.assert_close(x[:, 4608:], x[:, :1536]*x[:, 1536:3072], atol=0, rtol=0)
                probabilities = torch.nn.functional.linear(x, state['weight'], state['bias']).softmax(1)
                torch.testing.assert_close(probabilities, torch.tensor(case['probabilities']), atol=1e-7, rtol=1e-6)
                if old_case is not None:
                    oldprobs = torch.nn.functional.linear(x, oldstate['weight'], oldstate['bias']).softmax(1)
                    torch.testing.assert_close(oldprobs, torch.tensor(old_case['probabilities']), atol=1e-6, rtol=1e-5)
                    assert float((oldprobs-torch.tensor(old_case['probabilities'])).abs().max()) == case['original_head_consistency_max_delta']
                total_features += len(x)
            else:
                assert case['probabilities'] == [] and case['feature_file'] is None
            pair = load(Path(indexed[name]['path']))
            teacher, old = read_current_case(stage, entry, pins)
            assert teacher == pair['teacher']
            models = [teacher, pair['student'], pair['feature'], old['alternative'], *case['new_views']]
            for candidate in case['proposals']:
                voters = {}
                for model in models:
                    assert model['source_sha256'] == case['source_sha256']
                    for row in model['predictions']['merged_predictions']:
                        l, t, r, b = row['box_xyxy']
                        if row['class_id'] == candidate['class_id'] and row['confidence'] > .05 and 16 <= l < r <= 3632 and 16 <= t < b <= 2720:
                            value = iou(candidate['box_xyxy'], row['box_xyxy'])
                            if value >= .5:
                                key = model['weight_sha256']
                                voters[key] = max(value, voters.get(key, 0.))
                assert len(voters) == 3 and sorted(voters) == candidate['semantic_model_vote_sha256']
                assert voters == candidate['localization_voter_best_IoU']
            assert independent_selection(case['current'], case['proposals'], case['probabilities'], case['head_sha256']) == case['trial']
            gt = read_targets(stage, name, [2736, 3648], entry['label_sha256'], pins)
            a, b = matches(case['current']['all_predictions'], gt)[0], matches(case['trial']['all_predictions'], gt)[0]
            rows.append(dict(image=name, current=metric(case['current']['all_predictions'], gt),
                trial=metric(case['trial']['all_predictions'], gt), gained=sorted(b-a), lost=sorted(a-b), skip_reason=case['skip_reason']))
        assert rows == load(SOURCE / stage / 'report.json')['cases']
        totals = {v: {k: sum(r[v][k] for r in rows) for k in ('tp', 'unmatched', 'fn', 'predictions', 'targets')} for v in ('current', 'trial')}
        assert totals == report['stages'][stage]['summary']
        counts[stage], summaries[stage] = len(rows), totals
    assert all(sha(Path(p)) == v for p, v in report['pins'].items()) and sha(SOURCE / 'report.json') == digest
    OUT.mkdir()
    result = dict(status='pass', counts=counts, summaries=summaries, feature_candidates=total_features,
        source_report_sha256=digest, independent_cached_fresh_feature_head_vote_rank_GT_replay=True,
        DINO_extraction_not_independently_repeated=True, detector_reinference=False,
        no_training=True, no_deployment=True, field_accuracy=False, pins=pins)
    save(OUT / 'report.json', result)
    print({k: v for k, v in result.items() if k != 'pins'}, flush=True)


if __name__ == '__main__':
    main()
