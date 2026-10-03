"""TRAIN192 fresh exact-pose features, existing source-fold auxiliary heads."""
import copy
import os
import sys
import time
from pathlib import Path
sys.dont_write_bytecode = True
from prepare_paired_port_semantics import ROOT, REPO, DATA, OUT as ORIGINAL, load, save, sha, read_image, REFERENCE_SHA
from paired_port_semantics import expected_in_source, valid_boxes, paired_features
from port_semantic_verifier import embeddings
from current_port_baseline_audit import BASE, read_targets
from audit_port_multiscale_acceptance import metric, matches
from paired_pose_head_agreement import select
from inspection_agent.paired_port_geometry import HEAD_SHA
from inspection_agent.paired_median_geometry import median_runtime_fingerprint
SOURCE = ROOT / 'artifacts/paired_pose_search_20261003'
JITTER = ROOT / 'artifacts/paired_positive_jitter_20261003'
OUT = ROOT / 'artifacts/paired_pose_jitter_agreement_20261003'


def main():
    if OUT.exists(): raise FileExistsError('Preserve same-native pose agreement')
    import torch
    import cv2
    torch.set_num_threads(1); cv2.setNumThreads(1)
    source_report = load(SOURCE / 'report.json'); assert source_report['status'] == 'rejected'
    trained = load(JITTER / 'heads_oof/report.json'); assert trained['qualifies_crop_feasibility']
    prepared = load(JITTER / 'features_train/report.json')
    pins = dict(source_report['pins']); pins.update(trained['pins']); pins.update(trained['head_pins'])
    reference_path = DATA / 'images/train01/normal_073.JPG'; assert sha(reference_path) == REFERENCE_SHA
    for path in (Path(__file__), Path(__file__).with_name('paired_pose_head_agreement.py'),
        ROOT / 'artifacts/paired_pose_jitter_agreement_preregistration_20261003/PLAN.md',
        SOURCE / 'report.json', JITTER / 'heads_oof/report.json', JITTER / 'features_train/report.json', reference_path):
        pins[str(path)] = sha(path)
    assert {p:sha(Path(p)) for p in pins} == pins
    frozen = median_runtime_fingerprint(REPO); assert frozen == source_report['median_runtime_fingerprint']
    records = {r['image']:r for r in prepared['sources']}
    names = sorted(load(ROOT / 'artifacts/port_training_multiscale_20261002/protocol.json')['train_sources'])
    assert len(names) == 192 and all(records[name]['fold'] == i%3 for i,name in enumerate(names))
    head_path = REPO / 'output/paired_port_geometry_20261003/last_head.pt'
    assert sha(head_path) == HEAD_SHA
    original_head = torch.nn.Linear(6144,3)
    original_head.load_state_dict(torch.load(head_path,map_location='cpu',weights_only=True)['state_dict'])
    original_head.eval().requires_grad_(False); heads = {}
    for fold in range(3):
        path = JITTER / 'heads_oof' / f'fold{fold}_head.pt'
        head = torch.nn.Linear(6144,3); head.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
        heads[fold] = (head.eval().requires_grad_(False),sha(path))
    OUT.mkdir(); started = time.monotonic(); encoder = None; reference = read_image(reference_path)
    save(OUT / 'protocol.json',dict(pins=pins,median_runtime_fingerprint=frozen,
        classifier_source_excluded=True,YOLO_not_OOF=True,frozen_actual_poses=True,
        original_scores_not_boosted=True,GT_only_after_source_selection=True,no_deployment=True,field_accuracy=False))
    feature_arrays = []; metadata = []; measurements = []
    try:
        for index, entry in enumerate(load(BASE / 'train/report.json')['cases']):
            name = entry['image']; path = SOURCE / 'train' / (Path(name).stem + '_predictions.json')
            pins[str(path)] = sha(path); case = load(path); native = case['proposals']; folded = records[name]['fold']
            head, digest = heads[folded]; values = torch.empty((0,6144)); scores = []
            save(OUT / 'progress.json',dict(status='running',pid=os.getpid(),completed=index,total=192,
                image=name,phase='fresh_native_pose_embeddings',seconds=round(time.monotonic()-started,2)))
            if native:
                source = DATA / 'images/train01' / name; pins[str(source)] = sha(source)
                provenance = ORIGINAL / 'features_train' / (Path(name).stem + '_source.json')
                pins[str(provenance)] = sha(provenance); alignment_record = load(provenance)
                assert alignment_record['source_sha256'] == pins[str(source)]
                assert case['alignment'] == alignment_record['alignment']
                assert case['alignment']['alignment_quality']['reliable']
                image = read_image(source); expected, mask = expected_in_source(reference,
                    case['alignment']['source_to_reference_homography'],image.shape[:2])
                assert valid_boxes([r['box_xyxy'] for r in native],mask) == list(range(len(native)))
                if encoder is None:
                    import dino_feature_diff as dino
                    encoder = dino._model(); encoder.eval().requires_grad_(False); torch.set_num_threads(1)
                boxes = [r['box_xyxy'] for r in native]
                values = paired_features(embeddings(encoder,image,boxes),embeddings(encoder,expected,boxes))
                with torch.inference_mode():
                    repeat_old = original_head(values).softmax(dim=1)
                    assert torch.allclose(repeat_old,torch.tensor(case['probabilities']),atol=1e-5,rtol=1e-4), name
                    scores = head(values).softmax(dim=1).tolist()
            feature_arrays.append(values)
            metadata.extend(dict(image=name,fold=folded,proposal=copy.deepcopy(row)) for row in native)
            trial = select(case['current'],native,case['probabilities'],scores,HEAD_SHA,digest)
            save(OUT / (Path(name).stem + '_predictions.json'),dict(image=name,current=case['current'],trial=trial,
                proposals=native,original_probabilities=case['probabilities'],auxiliary_probabilities=scores,
                auxiliary_head_sha256=digest,source_classifier_fold=folded))
            targets = read_targets('train',name,[2736,3648],entry['label_sha256'],pins)
            old, new = case['current']['all_predictions'], trial['all_predictions']
            oh, nh = matches(old,targets)[0], matches(new,targets)[0]
            measurements.append(dict(image=name,current=metric(old,targets),trial=metric(new,targets),
                gained=sorted(nh-oh),lost=sorted(oh-nh),additions=len(trial['paired_semantic_additions'])))
        feature_path = OUT / 'native_features.pt'; torch.save(dict(features=torch.cat(feature_arrays)),feature_path)
        metadata_path = OUT / 'native_samples.json'; save(metadata_path,metadata)
        pins[str(feature_path)] = sha(feature_path); pins[str(metadata_path)] = sha(metadata_path)
        totals = {v:{k:sum(r[v][k] for r in measurements) for k in ('tp','unmatched','fn','predictions','targets')} for v in ('current','trial')}
        assert totals['current'] == source_report['stages']['train']['summary']['current']
        normal = sum(r['trial']['predictions'] for r in measurements if r['image'].startswith('normal_'))
        qualifies = totals['trial']['tp'] > 289 and totals['trial']['unmatched'] <= 4 and not any(r['lost'] for r in measurements) and normal == 0
        assert {p:sha(Path(p)) for p in pins} == pins and median_runtime_fingerprint(REPO) == frozen
        result = dict(status='complete',qualifies=qualifies,summary=totals,cases=measurements,pins=pins,
            native_pairs=len(metadata),classifier_OOF_only=True,YOLO_not_OOF=True,normal_cues=normal,
            native_feature_identity_reverified_against_original_head=True,seconds=round(time.monotonic()-started,2),
            requires_full_head_and_holdouts_if_pass=True,no_deployment=True,field_accuracy=False)
        save(OUT / 'report.json',result); save(OUT / 'progress.json',dict(status='complete',qualifies=qualifies,summary=totals))
        print(str(dict(qualifies=qualifies,summary=totals,seconds=result['seconds'])),flush=True)
    except BaseException as error:
        save(OUT / 'progress.json',dict(status='failed',error=type(error).__name__+': '+str(error))); raise


if __name__ == '__main__': main()
