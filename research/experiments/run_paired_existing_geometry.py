"""Finite existing-box coordinate experiment; old raw selections stay intact."""
import copy
import os
import sys
import time
from pathlib import Path
sys.dont_write_bytecode = True
from prepare_paired_port_semantics import ROOT, REPO, DATA, OUT as ORIGINAL, load, save, sha, read_image, cached_alignments, REFERENCE_SHA
from current_port_baseline_audit import BASE, read_current_case, read_targets
from paired_port_semantics import expected_in_source, valid_boxes, paired_features, CachedReferenceSIFT
from port_semantic_verifier import embeddings
from audit_port_multiscale_acceptance import metric, matches
from paired_existing_geometry import proposals, refine
from inspection_agent.paired_port_geometry import HEAD_SHA
from inspection_agent.paired_median_geometry import median_runtime_fingerprint
MODELS = ROOT / 'artifacts/paired_support_graph_20261003/source_selections'
MEDIAN = ROOT / 'artifacts/paired_median_current_head_20261003'
OUT = ROOT / 'artifacts/paired_existing_geometry_20261003'


def main():
    if OUT.exists(): raise FileExistsError('Preserve existing-geometry trial')
    import torch
    import cv2
    torch.set_num_threads(1); cv2.setNumThreads(1)
    frozen = median_runtime_fingerprint(REPO)
    pins = {str(p):sha(p) for p in (Path(__file__),Path(__file__).with_name('paired_existing_geometry.py'),
        ROOT / 'artifacts/paired_existing_geometry_preregistration_20261003/PLAN.md',
        REPO / 'inspection_agent/paired_median_geometry.py',REPO / 'inspection_agent/paired_port_median_features.py',
        REPO / 'config/paired_median_geometry_20261003.json')}
    weight = REPO / 'output/paired_port_geometry_20261003/last_head.pt'; assert sha(weight) == HEAD_SHA
    pins[str(weight)] = HEAD_SHA; head = torch.nn.Linear(6144,3)
    head.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True)['state_dict']); head.eval().requires_grad_(False)
    reference_path = DATA / 'images/train01/normal_073.JPG'; assert sha(reference_path) == REFERENCE_SHA
    pins[str(reference_path)] = sha(reference_path); reference = read_image(reference_path)
    alignments = cached_alignments(); encoder = None; OUT.mkdir(); inputs = OUT / 'proposals'; inputs.mkdir(); started = time.monotonic()
    # Generate every candidate before opening any labels or calculating outcomes.
    counts = {}
    for stage,count in (('train',192),('inner',48),('outer',30)):
        index_path = MODELS / stage / 'index.json'; pins[str(index_path)] = sha(index_path)
        indexed = {r['image']:r for r in load(index_path)['records']}; entries = load(BASE / stage / 'report.json')['cases']; assert len(entries) == count
        counts[stage] = 0
        for entry in entries:
            name = entry['image']; path = Path(indexed[name]['path']); assert sha(path) == indexed[name]['sha256']; pins[str(path)] = sha(path)
            models = load(path); teacher,old = read_current_case(stage,entry,pins); assert teacher == models['teacher']
            fixed = MEDIAN / stage / (Path(name).stem+'_predictions.json'); pins[str(fixed)] = sha(fixed); current = load(fixed)['trial']
            rows = proposals(teacher,[teacher,models['student'],models['feature'],old['alternative']],current)
            save(inputs / (stage+'_'+Path(name).stem+'.json'),dict(image=name,current=current,candidates=rows,GT_not_used=True))
            counts[stage] += len(rows)
    save(OUT / 'protocol.json',dict(pins=pins,median_runtime_fingerprint=frozen,counts=counts,GT_free_proposals=True,
        fixed_existing_prediction_count=True,new_hint_payload=False,head_sha256=HEAD_SHA,no_deployment=True,field_accuracy=False))
    import assembly_auto_review_robust_v3 as registration
    stages = {}
    try:
        with CachedReferenceSIFT(reference):
            for stage,count in (('train',192),('inner',48),('outer',30)):
                folder = OUT / stage; folder.mkdir(); rows = []
                entries = load(BASE / stage / 'report.json')['cases']; assert len(entries) == count
                for index,entry in enumerate(entries):
                    name = entry['image']; path = inputs / (stage+'_'+Path(name).stem+'.json'); pins[str(path)] = sha(path)
                    case = load(path); current = case['current']; native = case['candidates']; scores = []; alignment = None
                    save(OUT / 'progress.json',dict(status='running',pid=os.getpid(),stage=stage,image=name,completed=index,total=count,seconds=round(time.monotonic()-started,2)))
                    if native:
                        source = DATA / 'images' / ('val01' if stage=='outer' else 'train01') / name; pins[str(source)] = sha(source); image = read_image(source)
                        if stage == 'train':
                            provenance = ORIGINAL / 'features_train' / (Path(name).stem+'_source.json'); pins[str(provenance)] = sha(provenance); cached = load(provenance)
                            assert cached['source_sha256'] == pins[str(source)]; alignment = copy.deepcopy(cached['alignment'])
                        elif name in alignments:
                            provenance,cached = alignments[name]; pins[str(provenance)] = sha(provenance)
                            assert cached['image_fingerprints']['source_sha256'] == pins[str(source)]; alignment = copy.deepcopy(cached['alignment'])
                        else: cv2.setRNGSeed(0); _,alignment = registration.automatic_homography(reference,image)
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
                        else: native = []
                    trial = refine(current,native,scores,HEAD_SHA)
                    save(folder / (Path(name).stem+'_predictions.json'),dict(image=name,current=current,trial=trial,candidates=native,probabilities=scores,alignment=alignment))
                    targets = read_targets(stage,name,[2736,3648],entry['label_sha256'],pins)
                    oh,nh = matches(current['all_predictions'],targets)[0],matches(trial['all_predictions'],targets)[0]
                    rows.append(dict(image=name,current=metric(current['all_predictions'],targets),trial=metric(trial['all_predictions'],targets),
                        gained=sorted(nh-oh),lost=sorted(oh-nh),changed=len(trial['existing_geometry_changes'])))
                totals = {v:{k:sum(r[v][k] for r in rows) for k in ('tp','unmatched','fn','predictions','targets')} for v in ('current','trial')}
                assert totals['current'] == load(MEDIAN / stage / 'report.json')['summary']['trial']
                normal = sum(r['trial']['predictions'] for r in rows if r['image'].startswith('normal_'))
                gain = totals['trial']['tp'] > totals['current']['tp'] if stage=='train' else totals['trial']['tp'] >= totals['current']['tp']
                qualifies = gain and totals['trial']['unmatched'] <= totals['current']['unmatched'] and not any(r['lost'] for r in rows) and normal == 0
                assert {p:sha(Path(p)) for p in pins} == pins and median_runtime_fingerprint(REPO) == frozen
                save(folder / 'report.json',dict(status='complete',qualifies=qualifies,summary=totals,cases=rows,normal_cues=normal))
                stages[stage] = dict(qualifies=qualifies,summary=totals); print(str(dict(stage=stage,**stages[stage])),flush=True)
                if not qualifies:
                    save(OUT / 'report.json',dict(status='rejected',stage=stage,stages=stages,pins=pins,seconds=round(time.monotonic()-started,2),no_deployment=True))
                    save(OUT / 'progress.json',dict(status='rejected',stage=stage)); return
        save(OUT / 'report.json',dict(status='source_pass_requires_reference_ROI_SAM_Qt',stages=stages,pins=pins,seconds=round(time.monotonic()-started,2),no_deployment=True,field_accuracy=False))
        save(OUT / 'progress.json',dict(status='complete',stages=stages))
    except BaseException as error:
        save(OUT / 'progress.json',dict(status='failed',error=type(error).__name__+': '+str(error))); raise


if __name__ == '__main__': main()
