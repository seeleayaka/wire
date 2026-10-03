"""One finite source-fold classifier + stricter proposal support experiment."""
import os
import sys
import time
from pathlib import Path
sys.dont_write_bytecode = True
from prepare_paired_port_semantics import ROOT, REPO, DATA, load, save, sha, read_image, cached_alignments, REFERENCE_SHA
from current_port_baseline_audit import BASE, read_targets
from paired_pose_three_vote import select
from audit_port_multiscale_acceptance import metric, matches
from port_semantic_verifier import fit_head, embeddings
from paired_port_semantics import expected_in_source, valid_boxes, paired_features, CachedReferenceSIFT
from inspection_agent.paired_median_geometry import median_runtime_fingerprint
JITTER = ROOT / 'artifacts/paired_positive_jitter_20261003'
PAIRS = ROOT / 'artifacts/paired_pose_jitter_agreement_20261003'
INPUTS = ROOT / 'artifacts/paired_pose_search_inputs_20261003'
MEDIAN = ROOT / 'artifacts/paired_median_current_head_20261003'
OUT = ROOT / 'artifacts/paired_pose_jitter_three_20261003'


def evaluate_train(folder, pins, *, full_head=None, features=None, metadata=None, digest=None):
    folder.mkdir(); rows = []; offset = 0
    for entry in load(BASE / 'train/report.json')['cases']:
        name = entry['image']; path = PAIRS / (Path(name).stem + '_predictions.json')
        pins[str(path)] = sha(path); case = load(path); native = case['proposals']
        if full_head is None:
            scores = case['auxiliary_probabilities']; head_sha = case['auxiliary_head_sha256']
        else:
            import torch
            local = metadata[offset:offset+len(native)]
            assert [r['proposal'] for r in local] == native and all(r['image'] == name for r in local)
            with torch.inference_mode(): scores = full_head(features[offset:offset+len(native)]).softmax(dim=1).tolist()
            offset += len(native); head_sha = digest
        current = case['current']; trial = select(current,native,scores,head_sha)
        save(folder / path.name,dict(image=name,current=current,trial=trial,proposals=native,probabilities=scores,head_sha256=head_sha))
        targets = read_targets('train',name,[2736,3648],entry['label_sha256'],pins)
        oh,nh = matches(current['all_predictions'],targets)[0],matches(trial['all_predictions'],targets)[0]
        rows.append(dict(image=name,current=metric(current['all_predictions'],targets),trial=metric(trial['all_predictions'],targets),gained=sorted(nh-oh),lost=sorted(oh-nh)))
    if full_head is not None: assert offset == len(metadata) == len(features)
    return stage_report(folder,rows,pins,strict_gain=True)


def stage_report(folder, rows, pins, *, strict_gain):
    totals = {v:{k:sum(r[v][k] for r in rows) for k in ('tp','unmatched','fn','predictions','targets')} for v in ('current','trial')}
    gain = totals['trial']['tp'] > totals['current']['tp'] if strict_gain else totals['trial']['tp'] >= totals['current']['tp']
    normal = sum(r['trial']['predictions'] for r in rows if r['image'].startswith('normal_'))
    qualifies = gain and totals['trial']['unmatched'] <= totals['current']['unmatched'] and not any(r['lost'] for r in rows) and normal == 0
    assert {p:sha(Path(p)) for p in pins} == pins
    result = dict(status='complete',qualifies=qualifies,summary=totals,cases=rows,normal_cues=normal)
    save(folder / 'report.json',result); print(str(dict(stage=folder.name,qualifies=qualifies,summary=totals)),flush=True)
    return result


def main():
    if OUT.exists(): raise FileExistsError('Preserve finite changed-classifier trial')
    import torch
    import cv2
    torch.set_num_threads(1); cv2.setNumThreads(1)
    paired = load(PAIRS / 'report.json'); assert paired['status'] == 'complete'
    trained = load(JITTER / 'heads_oof/report.json'); assert trained['qualifies_crop_feasibility']
    pins = dict(paired['pins'])
    for path in (Path(__file__), Path(__file__).with_name('paired_pose_three_vote.py'),
        ROOT / 'artifacts/paired_pose_jitter_three_preregistration_20261003/PLAN.md',PAIRS / 'report.json'):
        pins[str(path)] = sha(path)
    assert {p:sha(Path(p)) for p in pins} == pins
    frozen = median_runtime_fingerprint(REPO); OUT.mkdir(); started = time.monotonic(); stages = {}
    save(OUT / 'protocol.json',dict(pins=pins,median_runtime_fingerprint=frozen,source_folds_only=True,
        unchanged_p98=True,min_distinct_detector_votes=3,one_full_head_if_source_pass=True,no_deployment=True,field_accuracy=False))
    def finish(status, stage=None):
        assert median_runtime_fingerprint(REPO) == frozen and {p:sha(Path(p)) for p in pins} == pins
        result = dict(status=status,stage=stage,stages=stages,pins=pins,median_runtime_fingerprint=frozen,
            seconds=round(time.monotonic()-started,2),no_deployment=True,field_accuracy=False)
        save(OUT / 'report.json',result); save(OUT / 'progress.json',dict(status=status,stage=stage))
    try:
        save(OUT / 'progress.json',dict(status='running',pid=os.getpid(),phase='cached_source_OOF'))
        result = evaluate_train(OUT / 'source_train_oof',pins); stages['source_train_oof'] = dict(qualifies=result['qualifies'],summary=result['summary'])
        if not result['qualifies']: finish('rejected','source_train_oof'); return
        save(OUT / 'progress.json',dict(status='running',pid=os.getpid(),phase='fixed400_full_head'))
        data = torch.load(JITTER / 'features_train/features.pt',map_location='cpu',weights_only=True)
        torch.set_num_threads(2); head = fit_head(data['features'],data['labels']); torch.set_num_threads(1)
        destination = OUT / 'full'; destination.mkdir(); weight = destination / 'last_head.pt'
        torch.save(dict(state_dict=head.state_dict(),input_dimensions=6144,classes=['other','unplugged_plug','unplugged_jack'],
            encoder_sha256='b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9'),weight)
        digest = sha(weight); pins[str(weight)] = digest
        native = torch.load(PAIRS / 'native_features.pt',map_location='cpu',weights_only=True)['features']
        metadata = load(PAIRS / 'native_samples.json')
        result = evaluate_train(OUT / 'full_train',pins,full_head=head,features=native,metadata=metadata,digest=digest)
        stages['full_train'] = dict(qualifies=result['qualifies'],summary=result['summary'])
        if not result['qualifies']: finish('rejected','full_train'); return
        evaluate_holdouts(head,digest,pins,stages,started)
        failed = next((stage for stage in ('inner','outer') if stage in stages and not stages[stage]['qualifies']),None)
        finish('rejected' if failed else 'source_pass_requires_reference_ROI_SAM_Qt',failed)
    except BaseException as error:
        save(OUT / 'progress.json',dict(status='failed',error=type(error).__name__+': '+str(error))); raise


def evaluate_holdouts(head,digest,pins,stages,started):
    import torch
    import cv2
    import assembly_auto_review_robust_v3 as registration
    reference_path = DATA / 'images/train01/normal_073.JPG'; assert sha(reference_path) == REFERENCE_SHA
    pins[str(reference_path)] = sha(reference_path); reference = read_image(reference_path)
    alignments = cached_alignments(); encoder = None
    with CachedReferenceSIFT(reference):
        for stage,count in (('inner',48),('outer',30)):
            folder = OUT / stage; folder.mkdir(); rows = []; entries = load(BASE / stage / 'report.json')['cases']; assert len(entries) == count
            for index,entry in enumerate(entries):
                name = entry['image']; stem = Path(name).stem; path = INPUTS / (stage+'_'+stem+'_proposals.json')
                pins[str(path)] = sha(path); case = load(path); current = case['current']
                native = [p for p in case['extended_candidates'] if len(set(p['semantic_model_vote_sha256'])) >= 3]
                scores = []; alignment = None
                save(OUT / 'progress.json',dict(status='running',pid=os.getpid(),stage=stage,image=name,completed=index,total=count,seconds=round(time.monotonic()-started,2)))
                if native:
                    source = DATA / 'images' / ('val01' if stage=='outer' else 'train01') / name
                    pins[str(source)] = sha(source); image = read_image(source)
                    if name in alignments:
                        provenance,cached = alignments[name]; pins[str(provenance)] = sha(provenance)
                        assert cached['image_fingerprints']['source_sha256'] == pins[str(source)]; alignment = cached['alignment']
                    else: cv2.setRNGSeed(0); _,alignment = registration.automatic_homography(reference,image)
                    if alignment.get('alignment_quality',{}).get('reliable'):
                        expected,mask = expected_in_source(reference,alignment['source_to_reference_homography'],image.shape[:2])
                        native = [native[i] for i in valid_boxes([p['box_xyxy'] for p in native],mask)]
                        if native:
                            if encoder is None:
                                import dino_feature_diff as dino
                                encoder = dino._model(); encoder.eval().requires_grad_(False); torch.set_num_threads(1)
                            boxes = [p['box_xyxy'] for p in native]; values = paired_features(embeddings(encoder,image,boxes),embeddings(encoder,expected,boxes))
                            with torch.inference_mode(): scores = head(values).softmax(dim=1).tolist()
                    else: native = []
                trial = select(current,native,scores,digest)
                fixed = MEDIAN / stage / (stem+'_predictions.json'); pins[str(fixed)] = sha(fixed); assert current == load(fixed)['trial']
                save(folder / (stem+'_predictions.json'),dict(image=name,current=current,trial=trial,proposals=native,probabilities=scores,alignment=alignment,head_sha256=digest))
                targets = read_targets(stage,name,[2736,3648],entry['label_sha256'],pins)
                oh,nh = matches(current['all_predictions'],targets)[0],matches(trial['all_predictions'],targets)[0]
                rows.append(dict(image=name,current=metric(current['all_predictions'],targets),trial=metric(trial['all_predictions'],targets),gained=sorted(nh-oh),lost=sorted(oh-nh)))
            result = stage_report(folder,rows,pins,strict_gain=stage=='inner')
            stages[stage] = dict(qualifies=result['qualifies'],summary=result['summary'])
            if not result['qualifies']: return


if __name__ == '__main__': main()
