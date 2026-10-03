"""Fresh rectangular DINO inputs over frozen train-only sample metadata."""
import argparse
import copy
import os
import sys
import time
from pathlib import Path
sys.dont_write_bytecode = True
from prepare_paired_port_semantics import ROOT, REPO, DATA, OUT as ORIGINAL, load, save, sha, read_image
from prepare_paired_semantic_geometry import NEW as GEOMETRY
from paired_port_semantics import expected_in_source, valid_boxes, paired_features
from paired_rectcontext_features import embeddings, crop_signature
from inspection_agent.paired_port_geometry import paired_runtime_fingerprint
from inspection_agent.resolution_loose_plug_support import resolution_runtime_fingerprint
OUT = ROOT / 'artifacts/paired_rectcontext_20261003'


def main(mode):
    destination = OUT / ('smoke' if mode == 'smoke' else 'features_train')
    if destination.exists(): raise FileExistsError('Preserve rectangular features')
    import torch
    import cv2
    import numpy as np
    sample_path = ROOT / 'artifacts/paired_shape_train_probe_20261003/samples.json'
    rows = load(sample_path); prepared = load(GEOMETRY / 'features_train/report.json')
    assert len(rows) == 4166 and sum(r['kind'] == 'gt_port' for r in rows) == 344
    assert prepared['no_validation_training'] and prepared['frozen_encoder_unchanged']
    groups = prepared['source_groups']; assert len(groups) == 192
    if mode == 'smoke': groups = [name for name in groups if any(r['image'] == name and r['kind'] == 'gt_port' for r in rows)][:2]
    else: assert load(OUT / 'smoke/report.json')['real_rectcontext_inference_passed']
    frozen = paired_runtime_fingerprint(REPO); original_runtime = resolution_runtime_fingerprint(REPO)
    assert original_runtime == prepared['runtime_fingerprint']
    pins = dict(prepared['pins'])
    for path in (Path(__file__), Path(__file__).with_name('paired_rectcontext_features.py'), sample_path,
        GEOMETRY / 'features_train/report.json', ROOT / 'artifacts/paired_rectcontext_preregistration_20261003/PLAN.md'):
        pins[str(path)] = sha(path)
    for name in groups:
        path = ORIGINAL / 'features_train' / (Path(name).stem + '_source.json'); pins[str(path)] = sha(path)
    assert {p: sha(Path(p)) for p in pins} == pins
    destination.mkdir(parents=True)
    save(destination / 'protocol.json', dict(pins=pins, accepted_paired_runtime=frozen, mode=mode,
        train_sources=groups, rectangular_pixel_contexts=[1.5, 3.], minimum_per_axis=4,
        original_square_context_coverage85_retained=True, feature_dimensions=6144,
        original_CLS_and_central4x4=True, reference_self_expected_expected=True,
        no_metadata_features=True, no_validation_training=True, no_deployment=True, field_accuracy=False))
    import dino_feature_diff as dino
    torch.set_num_threads(2); cv2.setNumThreads(2); model = dino._model(); model.eval().requires_grad_(False); torch.set_num_threads(2)
    encoder_path = REPO / 'models/dinov2/weights/dinov2_vits14_pretrain.pth'; assert sha(encoder_path) == prepared['encoder_sha256']
    reference = read_image(DATA / 'images/train01/normal_073.JPG'); started = time.monotonic()
    values = []; metadata = []; records = []; cache_pins = {}; unique = requested = differences = changed_pixels = 0
    try:
        for index, name in enumerate(groups):
            local = [copy.deepcopy(r) for r in rows if r['image'] == name]
            source = load(ORIGINAL / 'features_train' / (Path(name).stem + '_source.json')); record = copy.deepcopy(source)
            save(destination / 'progress.json', dict(status='running', pid=os.getpid(), completed=index, total=len(groups), image=name,
                samples=len(metadata), phase='fresh_rectangular_pixel_embeddings', seconds=round(time.monotonic() - started, 2)))
            if local:
                image_path = DATA / 'images/train01' / name; assert sha(image_path) == source['source_sha256']; image = read_image(image_path)
                assert source['alignment']['alignment_quality']['reliable']
                expected, valid = expected_in_source(reference, source['alignment']['source_to_reference_homography'], image.shape[:2])
                boxes = [r['box'] for r in local]; assert valid_boxes(boxes, valid) == list(range(len(boxes)))
                observed_audit = {}; expected_audit = {}
                observed = embeddings(model, image, boxes, audit=observed_audit)
                expected_vectors = embeddings(model, expected, boxes, audit=expected_audit)
                selves = [i for i, r in enumerate(local) if r['kind'] == 'reference_self']
                if selves: observed[selves] = expected_vectors[selves]
                vectors = paired_features(observed, expected_vectors)
                for i, row in enumerate(local):
                    if row['kind'] != 'synthetic_aspect': continue
                    center = lambda b: [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]
                    target = next((j for j, r in enumerate(local) if r['kind'] == 'gt_port' and np.allclose(center(r['box']), center(row['box']), atol=1e-7, rtol=0)), None)
                    if target is None: continue
                    if any(crop_signature(row['box'], s) != crop_signature(local[target]['box'], s) for s in (1.5, 3.)): changed_pixels += 1
                    if not torch.equal(vectors[i], vectors[target]): differences += 1
                unique += observed_audit['unique_crops'] + expected_audit['unique_crops']
                requested += observed_audit['requested_crops'] + expected_audit['requested_crops']
                record.update(status='features_ready', rectangular_observed=observed_audit, rectangular_expected=expected_audit)
            else: vectors = torch.empty((0, 6144)); record.update(status='no_valid_sample_exact_short_circuit')
            path = destination / (Path(name).stem + '_features.pt'); torch.save(dict(features=vectors, samples=local), path)
            cache_pins[str(path)] = sha(path); values.append(vectors); metadata.extend(local); records.append(record)
            save(destination / (Path(name).stem + '_source.json'), record)
        combined = torch.cat(values); labels = torch.tensor([r['label'] for r in metadata]); folds = torch.tensor([r['fold'] for r in metadata])
        assert combined.shape == (len(metadata), 6144) and torch.isfinite(combined).all()
        assert differences > 0 and changed_pixels > 0 and unique < requested
        assert {p: sha(Path(p)) for p in pins} == pins and {p: sha(Path(p)) for p in cache_pins} == cache_pins
        assert paired_runtime_fingerprint(REPO) == frozen and sha(encoder_path) == prepared['encoder_sha256']
        assert not any(p.requires_grad for p in model.parameters())
        torch.save(dict(features=combined, labels=labels, folds=folds), destination / 'features.pt'); save(destination / 'samples.json', metadata)
        result = dict(status='complete', mode=mode, samples=len(metadata), counts=torch.bincount(labels, minlength=3).tolist(),
            gt_targets=sum(r['kind'] == 'gt_port' for r in metadata), gt_valid=sum(r['kind'] == 'gt_port' for r in metadata),
            source_groups=groups, sources=records, pins=pins, cache_pins=cache_pins,
            aggregate_feature_sha256=sha(destination / 'features.pt'), samples_sha256=sha(destination / 'samples.json'),
            runtime_fingerprint=original_runtime, accepted_paired_runtime=frozen, encoder_sha256=prepared['encoder_sha256'],
            no_validation_training=True, frozen_encoder_unchanged=True, real_rectcontext_inference_passed=True,
            real_short_axis_descriptor_differences=differences, changed_short_axis_pixel_signatures=changed_pixels,
            unique_encoder_crops=unique, requested_encoder_crops=requested,
            seconds=round(time.monotonic() - started, 2), no_deployment=True, field_accuracy=False)
        save(destination / 'report.json', result); save(destination / 'progress.json', dict(status='complete', samples=len(metadata), seconds=result['seconds']))
        print(str({k: v for k, v in result.items() if k not in ('source_groups', 'sources', 'pins', 'cache_pins', 'runtime_fingerprint', 'accepted_paired_runtime')}), flush=True)
    except BaseException as error:
        save(destination / 'progress.json', dict(status='failed', error=type(error).__name__ + ': ' + str(error))); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--mode', choices=('smoke', 'full'), required=True); main(parser.parse_args().mode)
