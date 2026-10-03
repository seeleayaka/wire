"""Isolated capture and replay of pre-merge Mendeley validation evidence."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys


def setup(repo):
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / 'prototype'))
    import assembly_auto_review_dino_v2
    import tiled_dino_review
    return assembly_auto_review_dino_v2.implementation, tiled_dino_review


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def capture(args):
    impl, tiled = setup(args.repo)
    from evaluate_mendeley_balanced import FULL_REVIEW_ROI, read_image
    import assembly_auto_review_robust_v3 as perspective
    original_merge = tiled._merge_candidates
    original_review = impl.review_components
    trace = []
    raw = []

    def merge_hook(candidates):
        raw.append(copy.deepcopy(candidates))
        return original_merge(candidates)

    def review_hook(reference, aligned, valid, **kwargs):
        before = len(raw)
        result = original_review(reference, aligned, valid, **kwargs)
        if len(raw) != before + 1 or not kwargs['require_whole_cross_evidence']:
            raise RuntimeError('unexpected review path; capture invalid')
        trace.append({'raw': raw[-1], 'height': reference.shape[0], 'width': reference.shape[1],
                      'local_candidates': copy.deepcopy(result[2]), 'metadata': copy.deepcopy(result[1])})
        return result

    tiled._merge_candidates = merge_hook
    impl.review_components = review_hook
    tiled.LARGE_ROI_MAX_CANDIDATES = 6
    dataset = args.repo / 'data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults'
    reference = read_image(dataset / 'images/train01/normal_073.JPG')
    bounds = impl.adaptive.local_review.motion.perspective.auto.base.pixels(reference, FULL_REVIEW_ROI[0])
    expected_report = json.loads((args.repo / 'output/mendeley_reference_localization_20260927/val_fixed_faults_budget6.json').read_text(encoding='utf-8'))
    expected = {row['image']: row for row in expected_report['cases']}
    try:
        for index, name in enumerate(sorted(expected), 1):
            path = args.output / (Path(name).stem + '.json')
            trace.clear()
            raw.clear()
            inspection = read_image(dataset / 'images/val01' / name)
            aligned, alignment = perspective.automatic_homography(reference, inspection)
            if aligned is None:
                raise RuntimeError(f'alignment failed: {name}')
            _overlay, _heat, candidates = impl.dino_fused_regions(reference, aligned, FULL_REVIEW_ROI)
            coords = lambda rows: [[row[k] for k in ('left', 'top', 'right', 'bottom')] for row in rows]
            if coords(candidates) != coords(expected[name]['candidates']) or len(trace) != 1:
                raise RuntimeError(f'baseline mismatch: {name}')
            save(path, {'image': name, 'split': 'val01', 'bounds': bounds, 'trace': trace[0],
                        'candidates': candidates, 'targets': expected[name]['target_boxes_aligned_xyxy'],
                        'source_sha256': hashlib.sha256((args.repo / 'prototype/tiled_dino_review.py').read_bytes()).hexdigest()})
            print(f'{index}/{len(expected)} {name}: raw={len(raw[0])} published={len(candidates)} baseline_exact=True', flush=True)
    finally:
        tiled._merge_candidates = original_merge
        impl.review_components = original_review


def area(c):
    return max(0, c['right'] - c['left']) * max(0, c['bottom'] - c['top'])


def merge_variant(raw, tiled, mode):
    if mode == 'baseline':
        return tiled._merge_candidates(raw)
    def compatible(a, b):
        return (tiled._iou(a, b) >= .25 or tiled._contains(a, b) or tiled._contains(b, a)
                or (mode != 'overlap_only' and tiled._near_touching(a, b)))
    pending = copy.deepcopy(raw)
    groups = []
    while pending:
        group = [pending.pop(0)]
        changed = True
        while changed:
            changed = False
            kept = []
            for item in pending:
                members = group[:1] if mode == 'seed' else group
                accept = all(compatible(item, member) for member in members) if mode == 'complete' else any(compatible(item, member) for member in members)
                if accept and mode.startswith('growth_'):
                    proposed = group + [item]
                    union = {'left': min(x['left'] for x in proposed), 'top': min(x['top'] for x in proposed),
                             'right': max(x['right'] for x in proposed), 'bottom': max(x['bottom'] for x in proposed)}
                    accept = area(union) <= float(mode.split('_')[1]) * max(area(x) for x in proposed)
                if accept:
                    group.append(item)
                    changed = True
                else:
                    kept.append(item)
            pending = kept
        groups.append(group)
    merged = [box for group in groups for box in tiled._merge_candidates(group)]
    return sorted(merged, key=lambda x: (x['evidence_summary']['whole_roi_overlap'] or x['evidence_summary']['source_tile_count'] > 1,
                  not x['evidence_summary']['tile_edge_only'], x['area'] * x['difference_score']), reverse=True)


def publish(merged, trace, tiled, matched_budget=False):
    tiled._annotate_roi_edges(merged, trace['width'], trace['height'])
    merged, _ = tiled._publish_candidates(merged, [])
    merged, _ = tiled._display_candidates_for_large_roi(merged)
    budget, _ = tiled._large_roi_candidate_budget(merged)
    if matched_budget:
        budget = min(budget, len(trace['local_candidates']))
    published, _ = tiled._limit_candidates_for_large_roi(merged, budget=budget)
    return published


def replay(args):
    _, tiled = setup(args.repo)
    tiled.LARGE_ROI_MAX_CANDIDATES = 6
    from tools.evaluate_mendeley_local_refinement import summarize
    records = [json.loads(path.read_text(encoding='utf-8')) for path in sorted(args.output.glob('*.json')) if path.name != 'comparison.json']
    if len(records) != 15 or any(row['split'] != 'val01' for row in records):
        raise ValueError('requires all 15 validation records')
    current_hash = hashlib.sha256((args.repo / 'prototype/tiled_dino_review.py').read_bytes()).hexdigest()
    if any(row['source_sha256'] != current_hash for row in records):
        raise ValueError('source code changed since capture')
    results = []
    for mode in ('baseline', 'overlap_only', 'seed', 'complete', 'growth_1.5', 'growth_2', 'growth_3', 'growth_4', 'growth_6', 'growth_2_matched', 'growth_3_matched', 'seed_matched'):
        cases = []
        for record in records:
            trace = record['trace']
            if trace['metadata']['repetitive_group_candidate_count']:
                raise ValueError('unsupported repetitive groups')
            merged = merge_variant(trace['raw'], tiled, mode.removesuffix('_matched'))
            local = publish(merged, trace, tiled, matched_budget=mode.endswith('_matched'))
            if mode == 'baseline' and local != trace['local_candidates']:
                raise ValueError(f'local replay mismatch: {record["image"]}')
            left, top, _, _ = record['bounds']
            candidates = [{**box, 'left': box['left'] + left, 'right': box['right'] + left,
                           'top': box['top'] + top, 'bottom': box['bottom'] + top} for box in local]
            candidates.sort(key=lambda x: x['area'] * x['difference_score'], reverse=True)
            cases.append({'image': record['image'], 'targets': record['targets'], 'candidates': candidates,
                          'raw_count': len(trace['raw']), 'merged_count': len(merged)})
        metrics = summarize(cases, 'candidates')
        by_kind = {kind: summarize([case for case in cases if case['image'].startswith(kind + '_')], 'candidates')
                   for kind in ('damaged', 'disconnected', 'misrouted')}
        results.append({'mode': mode, 'metrics': metrics, 'by_kind': by_kind, 'cases': cases})
        print(json.dumps({'mode': mode, **metrics}), flush=True)
    save(args.output / 'comparison.json', {'split': 'val01', 'baseline_exact': True, 'results': results})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['capture', 'replay'])
    parser.add_argument('--repo', type=Path, default=Path('E:/PythonProject10'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    (capture if args.mode == 'capture' else replay)(args)
