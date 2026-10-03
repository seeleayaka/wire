"""Fixed spatial-normal evidence experiment; no production mutation or test tuning."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def normalized_features(value):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim not in (3, 4) or min(value.shape) == 0 or not np.isfinite(value).all():
        raise ValueError('invalid features')
    return value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-8)


def fit_spatial(bank, dimensions=32, seed=20260928, shrinkage=0.1, floor=1e-6):
    bank = normalized_features(bank)
    if bank.ndim != 4 or len(bank) < 2 or not 1 <= dimensions <= bank.shape[-1]:
        raise ValueError('invalid fitting bank or dimensions')
    if not 0 < shrinkage <= 1 or floor <= 0:
        raise ValueError('invalid covariance regularization')
    channels = np.sort(np.random.default_rng(seed).choice(bank.shape[-1], dimensions, replace=False))
    reduced = bank[..., channels]
    mean = reduced.mean(axis=0)
    delta = reduced - mean
    covariance = np.einsum('nhwc,nhwd->hwcd', delta, delta) / (len(bank) - 1)
    identity = np.eye(dimensions)
    isotropic = np.trace(covariance, axis1=-2, axis2=-1) / dimensions
    covariance = (1-shrinkage)*covariance + shrinkage*isotropic[..., None, None]*identity
    covariance += floor*identity
    return {'channels': channels, 'mean': mean, 'precision': np.linalg.inv(covariance)}


def spatial_score(query, model):
    query = normalized_features(query)
    if query.ndim != 3 or query.shape[:2] != model['mean'].shape[:2]:
        raise ValueError('query geometry mismatch')
    if query.shape[-1] <= int(model['channels'].max()):
        raise ValueError('query channel mismatch')
    delta = query[..., model['channels']] - model['mean']
    squared = np.einsum('hwc,hwcd,hwd->hw', delta, model['precision'], delta)
    return np.sqrt(np.maximum(squared, 0) / len(model['channels']))


def normal_scale(maps):
    values = np.array([np.percentile(m, 95) for m in maps], dtype=float)
    if len(values) < 2 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('invalid calibration maps')
    raw = float(np.percentile(values, 95))
    return max(raw, 1e-8), {'image_p95': values.tolist(), 'raw_scale': raw,
                           'floor_applied': raw < 1e-8}


def fuse(local, position, local_scale, position_scale):
    if local.shape != position.shape or min(local_scale, position_scale) <= 0:
        raise ValueError('invalid fusion geometry or scale')
    return np.maximum(local/local_scale, position/position_scale)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path('E:/PythonProject10'))
    parser.add_argument('--feature-cache', type=Path, required=True)
    parser.add_argument('--fault-cache', type=Path, required=True)
    parser.add_argument('--normal-cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('use a new experiment output directory')
    sys.path.insert(0, str(args.repo))
    from tools.merge_audit import setup, save
    from tools.probe_local_normal_bank import local_distance, region_score
    from tools.evaluate_anchored_local import anchored_selection
    from tools.evaluate_mendeley_local_refinement import summarize
    _, tiled = setup(args.repo)
    from evaluate_mendeley_balanced import read_image
    tiled.LARGE_ROI_MAX_CANDIDATES = 6
    dataset = args.repo/'data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults'
    previous = json.loads((args.feature_cache/'report.json').read_text(encoding='utf-8'))
    # The original coarse report predates the optional resolution flag; its
    # fixed default and grid fingerprint identify the 392 experiment.
    if previous.get('feature_max_edge', 392) != 392 or previous['grid'] != [21, 28, 384]:
        raise ValueError('expected the frozen coarse feature cache')
    records = []
    for directory in (args.fault_cache, args.normal_cache):
        records += [json.loads(p.read_text(encoding='utf-8')) for p in sorted(directory.glob('*.json'))
                    if p.name != 'comparison.json']
    assert len(records) == 30 and len({r['image'] for r in records}) == 30
    assert all(r['split'] == 'val01' for r in records)
    bounds = records[0]['bounds']
    assert all(r['bounds'] == bounds for r in records)
    source_hash = hashlib.sha256((args.repo/'prototype/tiled_dino_review.py').read_bytes()).hexdigest()
    assert all(r['source_sha256'] == source_hash for r in records)
    canonical = read_image(dataset/'images/train01/normal_073.JPG')
    canonical_hash = hashlib.sha256(canonical.tobytes()).hexdigest()
    grid = tuple(previous['grid'])
    feature_manifest = []

    def features(path):
        stat = path.stat()
        identity = hashlib.sha256((str(path.resolve())+str(stat.st_size)+str(stat.st_mtime_ns)
                                  +str(bounds)+str(grid)+canonical_hash+'local_bank_v1').encode()).hexdigest()
        cached = args.feature_cache/(identity+'.npz')
        if not cached.is_file():
            raise FileNotFoundError(f'matching feature cache missing for {path.name}')
        with np.load(cached, allow_pickle=False) as data:
            value = data['features']
        if value.shape != grid or not np.isfinite(value).all():
            raise ValueError(f'invalid cache for {path.name}')
        feature_manifest.append({'image':path.name, 'split':path.parent.name,
                                 'cache':cached.name, 'cache_sha256':hashlib.sha256(cached.read_bytes()).hexdigest()})
        return value

    normals = sorted((dataset/'images/train01').glob('normal_*.JPG'))
    assert len(normals) == 120 and [p.name for p in normals] == previous['bank_names']
    fitting, embargo, calibration = normals[:80], normals[80:90], normals[90:]
    assert 'normal_073.JPG' in {p.name for p in fitting}
    bank = np.stack([features(p) for p in fitting])
    position_model = fit_spatial(bank)
    calibration_maps = {'local':[], 'position':[]}
    for index, path in enumerate(calibration, 1):
        query = features(path)
        calibration_maps['local'].append(local_distance(query, bank, k=3, radius=1))
        calibration_maps['position'].append(spatial_score(query, position_model))
        print(f'calibration {index}/30 {path.name}', flush=True)
    local_scale, local_cal = normal_scale(calibration_maps['local'])
    position_scale, position_cal = normal_scale(calibration_maps['position'])
    fused_cal = [fuse(l, p, local_scale, position_scale)
                 for l, p in zip(calibration_maps['local'], calibration_maps['position'])]
    fusion_scale, fusion_cal = normal_scale(fused_cal)
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'spatial_model.npz', **position_model)
    np.savez_compressed(args.output/'calibration_maps.npz', local=np.stack(calibration_maps['local']),
                        position=np.stack(calibration_maps['position']), fusion=np.stack(fused_cal))
    cases = {m:[] for m in ('baseline', 'local80_anchor', 'position_anchor', 'fusion_anchor')}
    evidence = []
    geometry_changes = {m:0 for m in cases}
    for index, record in enumerate(records, 1):
        query = features(dataset/'images/val01'/record['image'])
        local = local_distance(query, bank, k=3, radius=1)
        position = spatial_score(query, position_model)
        combined = fuse(local, position, local_scale, position_scale)
        trace = record['trace']
        assert not trace['metadata']['repetitive_group_candidate_count']
        pool = tiled._merge_candidates(copy.deepcopy(trace['raw']))
        tiled._annotate_roi_edges(pool, trace['width'], trace['height'])
        published, _ = tiled._publish_candidates(pool, [])
        eligible, _ = tiled._display_candidates_for_large_roi(published)
        budget, _ = tiled._large_roi_candidate_budget(eligible)
        baseline, _ = tiled._limit_candidates_for_large_roi(eligible, budget=budget)
        assert baseline == trace['local_candidates']
        maps = {'local80_anchor':local, 'position_anchor':position, 'fusion_anchor':combined}
        coordinate_key = lambda boxes: {tuple(b[k] for k in ('left','top','right','bottom')) for b in boxes}
        for mode in cases:
            selected = baseline if mode == 'baseline' else anchored_selection(
                baseline, pool, maps[mode], trace['width'], trace['height'], region_score)
            assert len(selected) == len(baseline) and (not baseline or selected[0] == baseline[0])
            if record['image'].startswith('normal_'):
                geometry_changes[mode] += coordinate_key(selected) != coordinate_key(baseline)
            left, top, _, _ = bounds
            global_boxes = [{**b, 'left':b['left']+left, 'right':b['right']+left,
                             'top':b['top']+top, 'bottom':b['bottom']+top} for b in selected]
            cases[mode].append({'image':record['image'], 'targets':record['targets'], 'candidates':global_boxes})
        evidence.append({'image':record['image'], 'local_p95':float(np.percentile(local,95)),
                         'position_p95':float(np.percentile(position,95)),
                         'fusion_p95':float(np.percentile(combined,95)),
                         'local_normalized':float(np.percentile(local,95)/local_scale),
                         'position_normalized':float(np.percentile(position,95)/position_scale),
                         'fusion_normalized':float(np.percentile(combined,95)/fusion_scale)})
        np.savez_compressed(args.output/(Path(record['image']).stem+'_maps.npz'),
                            local=local, position=position, fusion=combined)
        print(f'validation {index}/30 {record["image"]}', flush=True)

    def metrics(rows):
        faults = [r for r in rows if not r['image'].startswith('normal_')]
        normal_rows = [r for r in rows if r['image'].startswith('normal_')]
        single = [summarize([r], 'candidates') for r in faults]
        return {'faults':summarize(faults,'candidates') if faults else None,
                'image_macro_overlap_fraction':float(np.mean([s['target_boxes_with_overlap']/s['target_boxes'] for s in single])) if single else None,
                'image_macro_best_iou':float(np.mean([s['mean_best_target_iou'] for s in single])) if single else None,
                'normal_candidate_count':sum(len(r['candidates']) for r in normal_rows)}

    result = {mode:{'overall':metrics(rows), 'by_kind':{
        kind:metrics([r for r in rows if r['image'].startswith(kind+'_')])
        for kind in ('damaged','disconnected','misrouted','normal')},
        'normal_geometry_changes':geometry_changes[mode], 'cases':rows,
        'by_image':[{'image':r['image'], **metrics([r])} for r in rows]}
        for mode, rows in cases.items()}
    baseline_metrics = result['baseline']['overall']
    gates = {}
    for mode, data in result.items():
        faults = data['overall']['faults']
        gates[mode] = {'image_coverage_not_worse':faults['fault_images_with_target_overlap'] >= baseline_metrics['faults']['fault_images_with_target_overlap'],
                      'macro_coverage_not_worse':data['overall']['image_macro_overlap_fraction'] >= baseline_metrics['image_macro_overlap_fraction'],
                      'each_class_coverage_not_worse':all(data['by_kind'][k]['faults']['target_boxes_with_overlap'] >= result['baseline']['by_kind'][k]['faults']['target_boxes_with_overlap']
                                                           for k in ('damaged','disconnected','misrouted'))}
    normal_responses = {}
    for branch in ('local','position','fusion'):
        normals_e = [e[branch+'_normalized'] for e in evidence if e['image'].startswith('normal_')]
        faults_e = [e[branch+'_normalized'] for e in evidence if not e['image'].startswith('normal_')]
        normal_responses[branch] = {'normal_images_above_calibration_scale':sum(v>1 for v in normals_e),
                                   'normal_image_count':len(normals_e), 'normal_median':float(np.median(normals_e)),
                                   'normal_max':float(max(normals_e)), 'fault_images_above_scale':sum(v>1 for v in faults_e),
                                   'warning':'Diagnostic threshold only, not deployed gate or measured field false-positive rate.'}
    save(args.output/'report.json', {'protocol':'spatial_normal_v1', 'split':'val01',
         'fit_names':[p.name for p in fitting], 'embargo_names':[p.name for p in embargo],
         'calibration_names':[p.name for p in calibration], 'dimensions':32,'seed':20260928,
         'shrinkage':0.1,'covariance_floor':1e-6,'feature_max_edge':392,'grid':list(grid),
         'local_k':3,'local_radius':1,'bounds':bounds,'source_sha256':source_hash,
         'experiment_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'feature_manifest':feature_manifest, 'legacy_feature_cache_warning':'Source cache identity omits model weight version; retained existing cache, fingerprints recorded.',
         'calibration':{'local_scale':local_scale,'position_scale':position_scale,'fusion_scale':fusion_scale,
                        'local':local_cal,'position':position_cal,'fusion':fusion_cal},
         'baseline_exact':True,'results':result,'evidence':evidence,'normal_responses':normal_responses,
         'development_gates':gates,'formal_path_changed':False,
         'warning':'Repeated validation development; correlated chassis scenes. Fixed normal candidate counts are not false-positive evidence. Max fusion is exploratory, not a faithful reproduction of cited papers.'})
    print(json.dumps({'overall':{m:r['overall'] for m,r in result.items()},
                      'gates':gates,'normal_responses':normal_responses}, indent=2))


if __name__ == '__main__':
    main()
