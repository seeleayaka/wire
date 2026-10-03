"""Single-image, default-off CNN review. Not a GUI or automatic fault verdict."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def review_image(image_path, model_path, train_normal_dir, *, experimental_cnn=False,
                 snapshot=None, config_dir=None, cascade=None, **options):
    if cascade is None:
        from tools import run_mendeley_review_cascade as cascade
    metadata = {'requested': experimental_cnn, 'status': 'disabled' if not experimental_cnn else 'skipped_upstream',
                'baseline_preserved': True}
    if not experimental_cnn:
        report = cascade.review_image(image_path, model_path, train_normal_dir, **options)
        return {**report, 'experimental_cnn': metadata}
    from tools.merge_audit import setup
    from inspection_agent.experimental_cnn import FrozenCnnEvidence, select_with_optional_cnn
    impl, tiled = setup(ROOT)
    original_fused = cascade.dino_fused_regions
    original_merge, original_review = tiled._merge_candidates, impl.review_components
    old_budget = tiled.LARGE_ROI_MAX_CANDIDATES
    raw, traces = [], []

    def merge_hook(rows):
        raw.append(copy.deepcopy(rows))
        return original_merge(rows)

    def review_hook(reference, aligned, valid, **kwargs):
        before = len(raw)
        result = original_review(reference, aligned, valid, **kwargs)
        traces.append({'raw': raw[-1] if len(raw) == before + 1 else None,
                       'height': reference.shape[0], 'width': reference.shape[1],
                       'local_candidates': copy.deepcopy(result[2]), 'metadata': copy.deepcopy(result[1]),
                       'cross_evidence': kwargs.get('require_whole_cross_evidence')})
        return result

    def fused_hook(reference, aligned, rois):
        nonlocal metadata
        raw.clear(); traces.clear()
        result = original_fused(reference, aligned, rois)  # Old pipeline errors must propagate.
        try:
            if options.get('candidate_budget', 6) != 6 or options.get('reference_mode', 'fixed') != 'fixed':
                raise ValueError('CNN supports frozen fixed-reference budget 6 only')
            if len(traces) != 1 or traces[0]['raw'] is None or not traces[0]['cross_evidence']:
                raise ValueError('unsupported candidate capture path')
            trace = traces[0]
            if trace['metadata']['repetitive_group_candidate_count']:
                raise ValueError('unsupported repetitive groups')
            bounds = impl.adaptive.local_review.motion.perspective.auto.base.pixels(reference, rois[0])
            pool = original_merge(copy.deepcopy(trace['raw']))
            tiled._annotate_roi_edges(pool, trace['width'], trace['height'])
            pool, suppressed = tiled._publish_candidates(pool, [])
            if suppressed: raise ValueError('unexpected repetitive suppression')
            eligible, _ = tiled._display_candidates_for_large_roi(pool)
            budget, _ = tiled._large_roi_candidate_budget(eligible)
            baseline, _ = tiled._limit_candidates_for_large_roi(eligible, budget=budget)
            if baseline != trace['local_candidates']: raise ValueError('baseline replay mismatch')
            evidence = FrozenCnnEvidence(ROOT, snapshot or ROOT/'output/mendeley_cnn_heat_evidence_20260929',
                                         config_dir or ROOT/'output/cnn_experimental_runtime_config')
            selected, metadata = select_with_optional_cnn(baseline, pool, trace['width'], trace['height'],
                    lambda: evidence.score(reference, aligned, bounds), enabled=True)
            if metadata['status'] != 'applied' or selected == baseline: return result
            # JSON entry point only: old overlay/heat are not published by this CLI.
            x, y, _, _ = bounds
            candidates = [{**box, 'left': box['left']+x, 'right': box['right']+x,
                           'top': box['top']+y, 'bottom': box['bottom']+y} for box in selected]
            candidates.sort(key=lambda box: box['area']*box['difference_score'], reverse=True)
            return result[0], result[1], candidates
        except Exception as error:
            metadata = {'requested': True, 'status': 'fallback', 'baseline_preserved': True,
                        'error_type': type(error).__name__, 'reason': str(error).split('\n')[0][:240]}
            return result

    tiled._merge_candidates, impl.review_components = merge_hook, review_hook
    cascade.dino_fused_regions = fused_hook
    try:
        report = cascade.review_image(image_path, model_path, train_normal_dir, **options)
        return {**report, 'experimental_cnn': metadata}
    finally:
        tiled._merge_candidates, impl.review_components = original_merge, original_review
        cascade.dino_fused_regions = original_fused
        tiled.LARGE_ROI_MAX_CANDIDATES = old_budget


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--train-normal-dir', type=Path, required=True)
    parser.add_argument('--reference-mode', choices=['fixed', 'nearest'], default='fixed')
    parser.add_argument('--fixed-reference', default='normal_073.JPG')
    parser.add_argument('--candidate-budget', type=int, default=6)
    parser.add_argument('--experimental-cnn', action='store_true')
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError('use a fresh report path')
    report = review_image(args.image, args.model, args.train_normal_dir,
             experimental_cnn=args.experimental_cnn, snapshot=args.snapshot,
             config_dir=args.output.parent/'ultralytics_config', reference_mode=args.reference_mode,
             fixed_reference=args.fixed_reference, candidate_budget=args.candidate_budget)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write('\n')
    print(json.dumps({'decision': report['decision'], 'candidate_count': len(report['candidates']),
                      'experimental_cnn': report['experimental_cnn'], 'output': str(args.output)}, ensure_ascii=False))


if __name__ == '__main__': main()
