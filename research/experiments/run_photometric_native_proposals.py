"""Finite paired-exposure candidate branch; original features/prefix preserved."""
import argparse
import copy
import hashlib
import os
import shutil
import sys
import time
from pathlib import Path
sys.dont_write_bytecode = True
from prepare_paired_port_semantics import ROOT, REPO, DATA, load, save, sha, read_image, cached_alignments, REFERENCE_SHA
from current_port_baseline_audit import BASE, read_current_case, read_targets
from inspection_agent.paired_native_pose import native_pose_runtime_fingerprint, HEAD_SHA, HEAD_RELATIVE
from inspection_agent.paired_native_pose_features import proposals, select
from inspection_agent.paired_port_features import expected_in_source, valid_boxes, embeddings, paired_features
from inspection_agent.optional_port_crop_review import CONFIG, predict
from inspection_agent.teacher_student_port_support import TEACHER_RELATIVE, TEACHER_SHA, STUDENT_RELATIVE, STUDENT_SHA
from inspection_agent.feature_residual_port_support import WEIGHT_RELATIVE, WEIGHT_SHA
from audit_port_multiscale_acceptance import metric, matches
from robust_port_exposure import compensate
from paired_port_semantics import CachedReferenceSIFT
OUT = ROOT / 'artifacts/photometric_native_proposals_20261004'
PLAN = ROOT / 'artifacts/photometric_native_proposals_preregistration_20261004/PLAN.md'
CURRENT = ROOT / 'artifacts/paired_pose_native_three_20261004'
PREP = ROOT / 'artifacts/paired_support_graph_20261003/source_selections'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('smoke', 'full'), required=True)
    mode = parser.parse_args().mode
    destination = OUT / mode
    if destination.exists():
        raise FileExistsError('Preserve photometric detector trial')
    estimator = ROOT / 'artifacts/port_exposure_invariance_replay_20261004/report.json'
    assert load(estimator)['status'] == 'pass'
    if mode == 'full':
        assert load(OUT / 'smoke/report.json')['status'] == 'complete'
    import psutil
    assert psutil.virtual_memory().available > 6 * 2**30, 'No competing memory-heavy model job'
    (destination / 'config/Ultralytics').mkdir(parents=True)
    shutil.copy2('C:/Windows/Fonts/arial.ttf', destination / 'config/Ultralytics/Arial.ttf')
    os.environ.update(YOLO_CONFIG_DIR=str(destination / 'config'), YOLO_OFFLINE='True',
                      YOLO_AUTOINSTALL='False', HF_HUB_OFFLINE='1')
    import cv2
    import torch
    from ultralytics import YOLO
    import assembly_auto_review_robust_v3 as registration
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    frozen = native_pose_runtime_fingerprint(REPO)
    config = copy.deepcopy(CONFIG)
    paths = [REPO / p for p in (TEACHER_RELATIVE, STUDENT_RELATIVE, WEIGHT_RELATIVE)]
    digests = [TEACHER_SHA, STUDENT_SHA, WEIGHT_SHA]
    assert len(set(digests)) == 3 and [sha(p) for p in paths] == digests
    headpath = REPO / HEAD_RELATIVE
    assert sha(headpath) == HEAD_SHA
    referencepath = DATA / 'images/train01/normal_073.JPG'
    assert sha(referencepath) == REFERENCE_SHA
    pins = {str(p): sha(p) for p in (Path(__file__), PLAN, estimator,
        Path(__file__).with_name('robust_port_exposure.py'), referencepath, headpath, *paths,
        REPO / 'inspection_agent/paired_native_pose_features.py',
        REPO / 'inspection_agent/paired_port_features.py',
        REPO / 'inspection_agent/optional_port_crop_review.py')}
    class Capped:
        def __init__(self, model): self.model = model
        def predict(self, *args, **kw):
            torch.set_num_threads(2)
            result = self.model.predict(*args, **kw)
            torch.set_num_threads(2)
            return result
    models = []
    for path in paths:
        model = YOLO(str(path))
        assert model.task == 'segment' and dict(model.names) == {0:'unplugged_plug', 1:'unplugged_jack'}
        models.append(Capped(model))
    head = torch.nn.Linear(6144, 3)
    head.load_state_dict(torch.load(headpath, map_location='cpu', weights_only=True)['state_dict'])
    head.eval().requires_grad_(False)
    encoder = None
    reference = read_image(referencepath)
    cached = cached_alignments()
    started = time.monotonic()
    stages = {}
    save(destination / 'protocol.json', dict(mode=mode, pins=pins, runtime=frozen,
        config=config, three_unique_weight_sha256=digests, source_features_original_pixels=True,
        transformed_input_separate_SHA=True, no_training=True, no_sweeps=True,
        current_baseline=[295, 68, 40], no_deployment=True, field_accuracy=False))
    try:
        with CachedReferenceSIFT(reference):
            for stage, count in (('train',192), ('inner',48), ('outer',30)):
                folder = destination / stage
                folder.mkdir()
                indexpath = PREP / stage / 'index.json'
                pins[str(indexpath)] = sha(indexpath)
                indexed = {r['image']:r for r in load(indexpath)['records']}
                entries = sorted(load(BASE / stage / 'report.json')['cases'], key=lambda r:r['image'])
                assert len(entries) == count
                prepared = []
                for entry in entries:
                    name = entry['image']
                    pairedpath = Path(indexed[name]['path'])
                    assert sha(pairedpath) == indexed[name]['sha256']
                    pins[str(pairedpath)] = sha(pairedpath)
                    paired = load(pairedpath)
                    teacher, _ = read_current_case(stage, entry, pins)
                    assert teacher == paired['teacher']
                    currentpath = CURRENT / ('full_train' if stage == 'train' else stage) / (Path(name).stem + '_predictions.json')
                    pins[str(currentpath)] = sha(currentpath)
                    current = load(currentpath)['trial']
                    remaining = 5 - (len(current['all_predictions']) - len(current['primary']))
                    assert remaining >= 0
                    prepared.append((entry, current, remaining))
                rows, inferred, valid_count, normal = [], 0, 0, 0
                for i, (entry, current, remaining) in enumerate(prepared):
                    name = entry['image']
                    source = DATA / 'images' / ('val01' if stage == 'outer' else 'train01') / name
                    pins[str(source)] = sha(source)
                    save(destination / 'progress.json', dict(status='running', pid=os.getpid(),
                        mode=mode, stage=stage, image=name, completed=i, total=count,
                        inferred=inferred, seconds=round(time.monotonic() - started, 2)))
                    native, scores, views, alignment = [], [], [], None
                    policy = dict(status='skipped', reason='shared_budget_full')
                    pixel_digest = None
                    if remaining:
                        if stage == 'train':
                            cachedpath = ROOT / 'artifacts/paired_port_semantics_20261003/features_train' / (Path(name).stem + '_source.json')
                            pins[str(cachedpath)] = sha(cachedpath)
                            record = load(cachedpath)
                            assert record['source_sha256'] == pins[str(source)]
                            alignment = copy.deepcopy(record['alignment'])
                        elif name in cached:
                            cachedpath, record = cached[name]
                            pins[str(cachedpath)] = sha(cachedpath)
                            assert record['image_fingerprints']['source_sha256'] == pins[str(source)]
                            alignment = copy.deepcopy(record['alignment'])
                        else:
                            cv2.setRNGSeed(0)
                            _, alignment = registration.automatic_homography(reference, read_image(source))
                        policy = dict(status='abstained', reason='unreliable_registration')
                        if alignment.get('alignment_quality', {}).get('reliable'):
                            image = read_image(source)
                            expected, mask = expected_in_source(reference, alignment['source_to_reference_homography'], image.shape[:2])
                            transformed, policy = compensate(image, expected, mask)
                            if policy['status'] == 'compensated':
                                pixel_digest = hashlib.sha256(transformed.tobytes()).hexdigest()
                                for model, digest in zip(models, digests):
                                    raw = predict(model, transformed)
                                    assert CONFIG == config and raw['source_shape'] == list(image.shape[:2])
                                    views.append(dict(image=name, source_sha256=pins[str(source)],
                                        source_SHA_is_original_coordinate_identity=True,
                                        input_pixels_sha256=pixel_digest, input_pixels_changed=True,
                                        weight_sha256=digest, predictions=raw))
                                inferred += 1
                                native = proposals(views[0], views, current)
                                native = [native[j] for j in valid_boxes([r['box_xyxy'] for r in native], mask)]
                                if native:
                                    if encoder is None:
                                        import dino_feature_diff as dino
                                        encoder = dino._model()
                                        encoder.eval().requires_grad_(False)
                                    torch.set_num_threads(2)
                                    boxes = [r['box_xyxy'] for r in native]
                                    # Important: the semantic gate does NOT see transformed image pixels.
                                    features = paired_features(embeddings(encoder, image, boxes), embeddings(encoder, expected, boxes))
                                    with torch.inference_mode(): scores = head(features).softmax(dim=1).tolist()
                    trial = select(current, native, scores, HEAD_SHA)
                    assert trial['all_predictions'][:len(current['all_predictions'])] == current['all_predictions']
                    valid_count += len(native)
                    save(folder / (Path(name).stem + '_predictions.json'), dict(image=name, current=current,
                        trial=trial, proposals=native, probabilities=scores, new_views=views,
                        alignment=alignment, photometric_policy=policy, input_pixels_sha256=pixel_digest))
                    assert native_pose_runtime_fingerprint(REPO) == frozen and CONFIG == config
                    if mode == 'smoke':
                        if inferred == 2: break
                        continue
                    targets = read_targets(stage, name, [2736,3648], entry['label_sha256'], pins)
                    oldhits = matches(current['all_predictions'], targets)[0]
                    newhits = matches(trial['all_predictions'], targets)[0]
                    rows.append(dict(image=name, current=metric(current['all_predictions'], targets),
                        trial=metric(trial['all_predictions'], targets), lost=sorted(oldhits-newhits), gained=sorted(newhits-oldhits)))
                    if name.startswith('normal_'): normal += len(trial['all_predictions'])
                if mode == 'smoke':
                    assert inferred == 2 and not any('/labels/' in p.replace('\\','/') for p in pins)
                    result = dict(status='complete', inferred=inferred, valid_proposals=valid_count,
                        pins=pins, runtime=frozen, accuracy_not_scored=True, no_deployment=True,
                        seconds=round(time.monotonic()-started, 2))
                    break
                totals = {version:{k:sum(row[version][k] for row in rows) for k in ('tp','unmatched','fn','predictions','targets')} for version in ('current','trial')}
                assert totals['current']['tp'] == {'train':295,'inner':68,'outer':40}[stage]
                qualifies = (totals['trial']['tp'] > totals['current']['tp'] if stage != 'outer' else totals['trial']['tp'] >= totals['current']['tp']) and totals['trial']['unmatched'] <= totals['current']['unmatched'] and normal == 0 and not any(row['lost'] for row in rows)
                stages[stage] = dict(qualifies=qualifies, summary=totals, inferred=inferred,
                    valid_proposals=valid_count, normal_cues=normal)
                save(folder / 'report.json', dict(status='complete', **stages[stage], cases=rows))
                print(dict(stage=stage, **stages[stage]), flush=True)
                if not qualifies: break
            if mode == 'full':
                failed = next((stage for stage, item in stages.items() if not item['qualifies']), None)
                result = dict(status='rejected' if failed else 'source_pass_requires_actual_reference_ROI_gates',
                    failed_stage=failed, stages=stages, pins=pins, runtime=frozen,
                    no_training=True, no_deployment=True, field_accuracy=False,
                    seconds=round(time.monotonic()-started, 2))
        assert all(sha(Path(p)) == digest for p, digest in pins.items())
        assert native_pose_runtime_fingerprint(REPO) == frozen and CONFIG == config
        save(destination / 'report.json', result)
        save(destination / 'progress.json', dict(status=result['status'], seconds=result['seconds']))
        print(dict(status=result['status'], seconds=result['seconds']), flush=True)
    except BaseException as error:
        save(destination / 'progress.json', dict(status='failed', error=type(error).__name__ + ': ' + str(error)))
        raise


if __name__ == '__main__':
    main()
