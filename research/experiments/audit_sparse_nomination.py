"""Independent sparse nomination replay, followed by TRAIN-only coverage."""
import copy
import math
import sys
from pathlib import Path
sys.dont_write_bytecode = True
from prepare_paired_port_semantics import ROOT, REPO, DATA, load, save, sha, read_image
from current_port_baseline_audit import BASE, read_targets
from audit_fine_consensus_rank import iou, duplicate
from audit_port_multiscale_acceptance import matches
from inspection_agent.paired_native_pose import HEAD_RELATIVE, HEAD_SHA, native_pose_runtime_fingerprint
from inspection_agent.paired_port_features import expected_in_source, valid_boxes

SOURCE = ROOT/'artifacts/sparse_nomination_20261004'
PREP = ROOT/'artifacts/two_vote_resolution_preparation_20261004'
OUT = ROOT/'artifacts/sparse_nomination_replay_20261004'


def independent_nominees(case):
    h, w = 2736, 3648
    pool = []
    for model in case['models']:
        assert model['source_sha256'] == case['source_sha256']
        assert model['predictions']['source_shape'] == [h, w]
        for row in model['predictions']['merged_predictions']:
            l, t, r, b = row['box_xyxy']
            assert all(math.isfinite(v) for v in (l, t, r, b))
            if row['confidence'] > .05 and 16 <= l < r <= w-16 and 16 <= t < b <= h-16:
                pool.append((model['weight_sha256'], row))
    assert len({m['weight_sha256'] for m in case['models']}) == 3
    pool.sort(key=lambda x: (-x[1]['confidence'], x[1]['class_id'], *x[1]['box_xyxy'], x[0]))
    representatives, selected = [], []
    for weight, p in pool:
        if any(p['class_id'] == q['class_id'] and iou(p['box_xyxy'], q['box_xyxy']) >= .5 for q in representatives):
            continue
        representatives.append(p)
        if duplicate(p['box_xyxy'], case['current']['all_predictions']):
            continue
        votes = sorted({v for v, q in pool if p['class_id'] == q['class_id'] and iou(p['box_xyxy'], q['box_xyxy']) >= .5})
        if len(votes) not in (1, 2):
            continue
        row = copy.deepcopy(p)
        row.update(semantic_model_vote_sha256=votes, nomination_checkpoint_sha256=weight,
                   action_nomination_only=True, automatic_fault_verdict=False)
        selected.append(row)
    return selected


def independent_actions(seeds, scores, remaining):
    ranked = []
    for seed, prob in zip(seeds, scores):
        assert len(prob) == 3 and all(math.isfinite(p) and 0 <= p <= 1 for p in prob)
        assert abs(sum(prob)-1) < 1e-5
        cls = max(range(3), key=lambda i: prob[i])
        if cls == seed['class_id']+1 and prob[cls] >= .98:
            row = copy.deepcopy(seed)
            row['nomination_probability'] = float(prob[cls])
            ranked.append(row)
    ranked.sort(key=lambda p: (-p['nomination_probability'], -p['confidence'], *p['box_xyxy'], p['class_id']))
    chosen = []
    for row in ranked:
        if len(chosen) >= remaining:
            break
        if not duplicate(row['box_xyxy'], chosen):
            chosen.append(row)
    return chosen


def main():
    import cv2
    import torch
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    if OUT.exists():
        raise FileExistsError('Preserve complete nomination audit')
    report = load(SOURCE/'report.json')
    assert report['status'] == 'complete' and report['no_GT_read'] and report['no_new_detector_inference']
    assert native_pose_runtime_fingerprint(REPO) == report['runtime']
    assert all(sha(Path(p)) == v for p, v in report['pins'].items())
    assert sha(REPO/HEAD_RELATIVE) == HEAD_SHA
    state = torch.load(REPO/HEAD_RELATIVE, map_location='cpu', weights_only=True)['state_dict']
    reference = read_image(DATA/'images/train01/normal_073.JPG')
    pins = {str(p): sha(p) for p in (Path(__file__), SOURCE/'report.json')}
    entries = {e['image']: e for e in load(BASE/'train/report.json')['cases']}
    rows, total_misses, covered_misses, precise_misses = [], 0, 0, 0
    for record in report['cases']:
        name, stem = record['image'], Path(record['image']).stem
        case = load(PREP/'train'/(stem+'_seeds.json'))
        saved = load(Path(record['path']))
        assert sha(Path(record['path'])) == record['sha256']
        assert saved['models'] == case['models'] and saved['current'] == case['current']
        assert saved['source_sha256'] == case['source_sha256']
        assert saved['remaining_budget'] == case['remaining_budget']
        seeds = independent_nominees(case) if case['remaining_budget'] else []
        assert len(seeds) == saved['raw_nominees'] == record['raw_nominees']
        if seeds:
            alignment = saved['alignment']
            if alignment.get('alignment_quality', {}).get('reliable'):
                _, mask = expected_in_source(reference, alignment['source_to_reference_homography'], (2736, 3648))
                seeds = [seeds[i] for i in valid_boxes([p['box_xyxy'] for p in seeds], mask)]
            else:
                seeds = []
        assert seeds == saved['seeds'] and len(seeds) == record['valid_nominees']
        scores = []
        if seeds:
            feature = torch.load(SOURCE/'train'/(stem+'_features.pt'), map_location='cpu', weights_only=True)
            x = feature['features']
            assert x.shape == (len(seeds), 6144) and torch.isfinite(x).all()
            assert feature['boxes'] == [p['box_xyxy'] for p in seeds] and feature['source_sha256'] == case['source_sha256']
            with torch.inference_mode():
                values = torch.nn.functional.linear(x, state['weight'], state['bias']).softmax(1)
            torch.testing.assert_close(values, torch.tensor(saved['probabilities']), atol=1e-7, rtol=1e-6)
            scores = values.tolist()
        assert len(scores) == len(saved['probabilities'])
        chosen = independent_actions(seeds, scores, case['remaining_budget'])
        assert chosen == saved['chosen_seeds'] and len(chosen) == record['actions']
        # GT is read AFTER reconstructing the label-free action policy.
        gt = read_targets('train', name, [2736, 3648], entries[name]['label_sha256'], pins)
        hit = matches(case['current']['all_predictions'], gt)[0]
        details = []
        for ti, target in enumerate(gt):
            if ti in hit:
                continue
            total_misses += 1
            exact = any(p['class_id'] == target['class_id'] and iou(p['box_xyxy'], target['box']) >= .5 for p in chosen)
            covered = False
            for p in chosen:
                l, t, r, b = p['box_xyxy']
                for delta in (-120, 120):
                    x = max(0, min(3648-960, round((l+r)/2+delta-480)))
                    y = max(0, min(2736-960, round((t+b)/2+delta-480)))
                    a, c, d, e = target['box']
                    if x+16 < a < d < x+960-16 and y+16 < c < e < y+960-16:
                        covered = True
            covered_misses += covered
            precise_misses += exact
            details.append(dict(target_index=ti, exact_nomination_match=exact, inside_nomination_ROI_interior=covered))
        rows.append(dict(image=name, nominees=len(seeds), actions=len(chosen), residual_targets=details))
    assert len(rows) == 192 and total_misses == 46
    assert sum(r['nominees'] for r in rows) == report['valid_nominees']
    assert sum(r['actions'] for r in rows) == report['actions']
    assert all(sha(Path(p)) == v for p, v in report['pins'].items())
    assert all(sha(Path(p)) == v for p, v in pins.items())
    OUT.mkdir()
    result = dict(status='pass', sources=192, source_report_sha256=sha(SOURCE/'report.json'),
                  nominees=report['valid_nominees'], actions=report['actions'],
                  residual_targets=46, residual_targets_inside_action_ROI=covered_misses,
                  residual_targets_precise_nomination_match=precise_misses,
                  next_detector_trial_has_geometric_potential=covered_misses > 0,
                  nomination_coverage_not_accuracy=True, no_validation_read=True,
                  fresh_DINO_not_independently_reextracted=True, no_deployment=True, cases=rows, pins=pins)
    save(OUT/'report.json', result)
    print({k: v for k, v in result.items() if k not in ('pins', 'cases')}, flush=True)


if __name__ == '__main__':
    main()
