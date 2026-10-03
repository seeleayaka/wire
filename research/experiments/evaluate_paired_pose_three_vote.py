"""Frozen TRAIN prediction reuse, fresh held-out pairs, one stricter vote gate."""
import copy
import os
import sys
import time
from pathlib import Path
sys.dont_write_bytecode = True
from prepare_paired_port_semantics import ROOT, REPO, DATA, load, save, sha, read_image, cached_alignments, REFERENCE_SHA
from current_port_baseline_audit import BASE, read_targets
from paired_port_semantics import expected_in_source, valid_boxes, paired_features, CachedReferenceSIFT
from port_semantic_verifier import embeddings
from audit_port_multiscale_acceptance import metric, matches
from paired_pose_three_vote import select
from inspection_agent.paired_port_geometry import HEAD_SHA
from inspection_agent.paired_median_geometry import median_runtime_fingerprint
SOURCE = ROOT / 'artifacts/paired_pose_search_20261003'
INPUTS = ROOT / 'artifacts/paired_pose_search_inputs_20261003'
MEDIAN = ROOT / 'artifacts/paired_median_current_head_20261003'
OUT = ROOT / 'artifacts/paired_pose_three_vote_20261003'


def main():
    if OUT.exists(): raise FileExistsError('Preserve stricter support trial')
    import torch
    import cv2
    torch.set_num_threads(1); cv2.setNumThreads(1)
    old_report = load(SOURCE / 'report.json'); assert old_report['status'] == 'rejected'
    frozen = median_runtime_fingerprint(REPO); assert frozen == old_report['median_runtime_fingerprint']
    pins = dict(old_report['pins'])
    for path in (Path(__file__), Path(__file__).with_name('paired_pose_three_vote.py'),
        ROOT / 'artifacts/paired_pose_three_vote_preregistration_20261003/PLAN.md',SOURCE / 'report.json'):
        pins[str(path)] = sha(path)
    assert {p:sha(Path(p)) for p in pins} == pins
    head_path = REPO / 'output/paired_port_geometry_20261003/last_head.pt'; assert sha(head_path) == HEAD_SHA
    head = torch.nn.Linear(6144,3); head.load_state_dict(torch.load(head_path,map_location='cpu',weights_only=True)['state_dict'])
    head.eval().requires_grad_(False)
    reference_path = DATA / 'images/train01/normal_073.JPG'; assert sha(reference_path) == REFERENCE_SHA
    pins[str(reference_path)] = sha(reference_path); reference = read_image(reference_path)
    alignments = cached_alignments(); encoder = None; OUT.mkdir(); started = time.monotonic(); stages = {}
    save(OUT / 'protocol.json',dict(pins=pins,median_runtime_fingerprint=frozen,
        train_frozen_probabilities=True,heldout_fresh_pair_inference=True,min_distinct_model_votes=3,
        old_prefix_unchanged=True,no_deployment=True,field_accuracy=False))
    import assembly_auto_review_robust_v3 as registration
    try:
        with CachedReferenceSIFT(reference):
            for stage,count in (('train',192),('inner',48),('outer',30)):
                folder = OUT / stage; folder.mkdir(); entries = load(BASE / stage / 'report.json')['cases']; assert len(entries) == count
                rows = []
                for index,entry in enumerate(entries):
                    name = entry['image']; stem = Path(name).stem
                    save(OUT / 'progress.json',dict(status='running',pid=os.getpid(),stage=stage,
                        image=name,completed=index,total=count,seconds=round(time.monotonic()-started,2)))
                    if stage == 'train':
                        path = SOURCE / stage / (stem + '_predictions.json'); pins[str(path)] = sha(path)
                        case = load(path); current,native,scores = case['current'],case['proposals'],case['probabilities']
                        status = 'verified_frozen_train_probabilities'; alignment = case['alignment']
                    else:
                        path = INPUTS / (stage + '_' + stem + '_proposals.json'); pins[str(path)] = sha(path)
                        case = load(path); current = case['current']
                        native = [p for p in case['extended_candidates'] if len(set(p['semantic_model_vote_sha256'])) >= 3]
                        scores = []; alignment = None; status = 'no_three_vote_candidate'
                        if native:
                            source = DATA / 'images' / ('val01' if stage == 'outer' else 'train01') / name
                            pins[str(source)] = sha(source); image = read_image(source)
                            if name in alignments:
                                provenance,cached = alignments[name]; pins[str(provenance)] = sha(provenance)
                                assert cached['image_fingerprints']['source_sha256'] == pins[str(source)]
                                alignment = copy.deepcopy(cached['alignment'])
                            else:
                                cv2.setRNGSeed(0); _,alignment = registration.automatic_homography(reference,image)
                            if alignment.get('alignment_quality',{}).get('reliable'):
                                expected,mask = expected_in_source(reference,alignment['source_to_reference_homography'],image.shape[:2])
                                native = [native[i] for i in valid_boxes([p['box_xyxy'] for p in native],mask)]
                                if native:
                                    if encoder is None:
                                        import dino_feature_diff as dino
                                        encoder = dino._model(); encoder.eval().requires_grad_(False); torch.set_num_threads(1)
                                    boxes = [p['box_xyxy'] for p in native]
                                    values = paired_features(embeddings(encoder,image,boxes),embeddings(encoder,expected,boxes))
                                    with torch.inference_mode(): scores = head(values).softmax(dim=1).tolist()
                                status = 'fresh_paired_three_vote_inference'
                            else: native = []; status = 'registration_abstention'
                    fixed = MEDIAN / stage / (stem + '_predictions.json'); pins[str(fixed)] = sha(fixed)
                    assert current == load(fixed)['trial']
                    trial = select(current,native,scores,HEAD_SHA)
                    save(folder / (stem + '_predictions.json'),dict(image=name,current=current,trial=trial,
                        proposals=native,probabilities=scores,alignment=alignment,feature_status=status))
                    targets = read_targets(stage,name,[2736,3648],entry['label_sha256'],pins)
                    oh,nh = matches(current['all_predictions'],targets)[0],matches(trial['all_predictions'],targets)[0]
                    rows.append(dict(image=name,current=metric(current['all_predictions'],targets),trial=metric(trial['all_predictions'],targets),
                        gained=sorted(nh-oh),lost=sorted(oh-nh),additions=len(trial['paired_semantic_additions']),feature_status=status))
                totals = {v:{k:sum(r[v][k] for r in rows) for k in ('tp','unmatched','fn','predictions','targets')} for v in ('current','trial')}
                assert totals['current'] == load(MEDIAN / stage / 'report.json')['summary']['trial']
                normal = sum(r['trial']['predictions'] for r in rows if r['image'].startswith('normal_'))
                gain = totals['trial']['tp'] > totals['current']['tp'] if stage != 'outer' else totals['trial']['tp'] >= totals['current']['tp']
                qualifies = gain and totals['trial']['unmatched'] <= totals['current']['unmatched'] and not any(r['lost'] for r in rows) and normal == 0
                save(folder / 'report.json',dict(status='complete',qualifies=qualifies,summary=totals,cases=rows,normal_cues=normal))
                stages[stage] = dict(qualifies=qualifies,summary=totals)
                print(str(dict(stage=stage,**stages[stage])),flush=True)
                assert {p:sha(Path(p)) for p in pins} == pins and median_runtime_fingerprint(REPO) == frozen
                if not qualifies:
                    save(OUT / 'report.json',dict(status='rejected',stage=stage,stages=stages,pins=pins,seconds=round(time.monotonic()-started,2),no_deployment=True))
                    save(OUT / 'progress.json',dict(status='rejected',stage=stage)); return
        save(OUT / 'report.json',dict(status='source_pass_requires_reference_ROI_SAM_Qt',stages=stages,pins=pins,
            median_runtime_fingerprint=frozen,seconds=round(time.monotonic()-started,2),no_deployment=True,field_accuracy=False))
        save(OUT / 'progress.json',dict(status='complete',stages=stages))
    except BaseException as error:
        save(OUT / 'progress.json',dict(status='failed',error=type(error).__name__+': '+str(error))); raise


if __name__ == '__main__': main()
