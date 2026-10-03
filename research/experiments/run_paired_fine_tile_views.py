"""Finite upstream fine-tile candidate probe; stronger default-off prefix."""
import argparse
import copy
import os
import shutil
import sys
import time
from pathlib import Path
sys.dont_write_bytecode = True
from prepare_paired_port_semantics import ROOT, REPO, DATA, OUT as ORIGINAL, load, save, sha, read_image, cached_alignments, REFERENCE_SHA
from current_port_baseline_audit import BASE, read_current_case, read_targets
from paired_fine_tile_views import infer
from paired_median_proposals import proposals
from inspection_agent.paired_port_features import select
from paired_port_semantics import expected_in_source,valid_boxes,paired_features,CachedReferenceSIFT
from port_semantic_verifier import embeddings
from inspection_agent.port_tiling import box_iou
from inspection_agent.paired_port_geometry import HEAD_SHA
from inspection_agent.paired_median_geometry import median_runtime_fingerprint
from inspection_agent.teacher_student_port_support import TEACHER_RELATIVE,TEACHER_SHA,STUDENT_RELATIVE,STUDENT_SHA
from inspection_agent.optional_port_crop_review import CONFIG
from audit_port_multiscale_acceptance import metric,matches
MODELS = ROOT / 'artifacts/paired_support_graph_20261003/source_selections'
MEDIAN = ROOT / 'artifacts/paired_median_current_head_20261003'
OUT = ROOT / 'artifacts/paired_fine_tile_views_20261003'


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--mode',choices=['smoke','full'],required=True); mode = parser.parse_args().mode
    destination = OUT / mode
    if destination.exists(): raise FileExistsError('Preserve finer-scale candidate trial')
    if mode == 'full': assert load(OUT / 'smoke/report.json')['status'] == 'complete'
    (destination / 'config/Ultralytics').mkdir(parents=True)
    shutil.copy2('C:/Windows/Fonts/arial.ttf',destination / 'config/Ultralytics/Arial.ttf')
    os.environ.update(YOLO_CONFIG_DIR=str(destination / 'config'),YOLO_OFFLINE='True',YOLO_AUTOINSTALL='False',HF_HUB_OFFLINE='1')
    import torch
    import cv2
    from ultralytics import YOLO
    torch.set_num_threads(4); cv2.setNumThreads(2)
    frozen = median_runtime_fingerprint(REPO); config = copy.deepcopy(CONFIG)
    pins = {str(p):sha(p) for p in (Path(__file__),Path(__file__).with_name('paired_fine_tile_views.py'),
        Path(__file__).with_name('paired_median_proposals.py'),ROOT / 'artifacts/paired_fine_tile_preregistration_20261003/PLAN.md')}
    paths = [REPO / TEACHER_RELATIVE, REPO / STUDENT_RELATIVE]; digests = [TEACHER_SHA,STUDENT_SHA]
    assert [sha(p) for p in paths] == digests; pins.update({str(p):d for p,d in zip(paths,digests)})
    models = [YOLO(str(p)) for p in paths]
    assert all(m.task == 'segment' and dict(m.names) == {0:'unplugged_plug',1:'unplugged_jack'} for m in models)
    class Capped:
        def __init__(self,model): self.model = model
        def predict(self,*args,**kwargs):
            torch.set_num_threads(4); result = self.model.predict(*args,**kwargs); torch.set_num_threads(4); return result
    wrapped = [Capped(m) for m in models]
    weight = REPO / 'output/paired_port_geometry_20261003/last_head.pt'; assert sha(weight) == HEAD_SHA
    pins[str(weight)] = HEAD_SHA; head = torch.nn.Linear(6144,3)
    head.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True)['state_dict']); head.eval().requires_grad_(False)
    reference_path = DATA / 'images/train01/normal_073.JPG'; assert sha(reference_path) == REFERENCE_SHA
    pins[str(reference_path)] = sha(reference_path); reference = read_image(reference_path)
    alignments = cached_alignments(); encoder = None; stages = {}; started = time.monotonic()
    save(destination / 'protocol.json',dict(pins=pins,median_runtime_fingerprint=frozen,mode=mode,
        original_config=config,tile_size=960,tile_stride=720,unchanged_model_weights=True,
        same_weight_views_one_vote=True,GT_after_predictions=True,no_deployment=True,field_accuracy=False))
    import assembly_auto_review_robust_v3 as registration
    try:
        with CachedReferenceSIFT(reference):
            for stage,count in (('train',192),('inner',48),('outer',30)):
                folder = destination / stage; folder.mkdir(); index_path = MODELS / stage / 'index.json'; pins[str(index_path)] = sha(index_path)
                indexed = {r['image']:r for r in load(index_path)['records']}; entries = load(BASE / stage / 'report.json')['cases']; assert len(entries) == count
                selected = []
                for entry in entries:
                    name = entry['image']; cache_path = Path(indexed[name]['path']); assert sha(cache_path) == indexed[name]['sha256']; pins[str(cache_path)] = sha(cache_path)
                    case = load(cache_path); teacher,old = read_current_case(stage,entry,pins); assert teacher == case['teacher']
                    fixed = MEDIAN / stage / (Path(name).stem+'_predictions.json'); pins[str(fixed)] = sha(fixed); current = load(fixed)['trial']
                    pool = [teacher,case['student'],case['feature'],old['alternative']]
                    remaining = 5-(len(current['all_predictions'])-len(current['primary'])); assert remaining >= 0
                    eligible = remaining > 0 and any(p['confidence'] > .05 for model in pool for p in model['predictions']['merged_predictions'])
                    selected.append((entry,current,pool,eligible))
                if mode == 'smoke': selected = sorted((r for r in selected if r[3] and r[1]['all_predictions']),key=lambda r:r[0]['image'])[:2]; assert len(selected) == 2
                rows = []; inferred = 0
                for index,(entry,current,pool,eligible) in enumerate(selected):
                    name = entry['image']; source = DATA / 'images' / ('val01' if stage=='outer' else 'train01') / name
                    pins[str(source)] = sha(source); native = []; scores = []; new_views = []; alignment = None
                    save(destination / 'progress.json',dict(status='running',pid=os.getpid(),stage=stage,image=name,completed=index,total=len(selected),inferred=inferred,seconds=round(time.monotonic()-started,2)))
                    if eligible:
                        image = read_image(source)
                        for model,digest in zip(wrapped,digests):
                            raw = infer(model,image); assert CONFIG == config and raw['source_shape'] == list(image.shape[:2])
                            new_views.append(dict(image=name,source_sha256=pins[str(source)],weight_sha256=digest,
                                inference_imgsz=960,tile_size=960,tile_stride=720,predictions=raw))
                        inferred += 1
                        native = proposals(pool[0],pool+new_views)
                        native = [p for p in native if not any(box_iou(p['box_xyxy'],q['box_xyxy']) >= .5 for q in current['all_predictions'])]
                        if stage == 'train':
                            provenance = ORIGINAL / 'features_train' / (Path(name).stem+'_source.json'); pins[str(provenance)] = sha(provenance); cached = load(provenance)
                            assert cached['source_sha256'] == pins[str(source)]; alignment = copy.deepcopy(cached['alignment'])
                        elif name in alignments:
                            provenance,cached = alignments[name]; pins[str(provenance)] = sha(provenance); assert cached['image_fingerprints']['source_sha256'] == pins[str(source)]; alignment = copy.deepcopy(cached['alignment'])
                        else: cv2.setRNGSeed(0); _,alignment = registration.automatic_homography(reference,image)
                        if alignment.get('alignment_quality',{}).get('reliable'):
                            expected,mask = expected_in_source(reference,alignment['source_to_reference_homography'],image.shape[:2])
                            native = [native[i] for i in valid_boxes([p['box_xyxy'] for p in native],mask)]
                            if native:
                                if encoder is None:
                                    import dino_feature_diff as dino
                                    encoder = dino._model(); encoder.eval().requires_grad_(False)
                                torch.set_num_threads(2); boxes = [p['box_xyxy'] for p in native]
                                values = paired_features(embeddings(encoder,image,boxes),embeddings(encoder,expected,boxes))
                                with torch.inference_mode(): scores = head(values).softmax(dim=1).tolist()
                        else: native = []
                    trial = select(current,native,scores,HEAD_SHA)
                    save(folder / (Path(name).stem+'_predictions.json'),dict(image=name,current=current,trial=trial,
                        proposals=native,probabilities=scores,new_views=new_views,alignment=alignment,eligible=eligible))
                    if mode == 'full':
                        targets = read_targets(stage,name,[2736,3648],entry['label_sha256'],pins)
                        oh,nh = matches(current['all_predictions'],targets)[0],matches(trial['all_predictions'],targets)[0]
                        rows.append(dict(image=name,current=metric(current['all_predictions'],targets),trial=metric(trial['all_predictions'],targets),gained=sorted(nh-oh),lost=sorted(oh-nh)))
                assert {p:sha(Path(p)) for p in pins} == pins and median_runtime_fingerprint(REPO) == frozen and CONFIG == config
                if mode == 'smoke':
                    save(destination / 'report.json',dict(status='complete',images=[r[0]['image'] for r in selected],inferred=inferred,pins=pins,seconds=round(time.monotonic()-started,2),accuracy_not_scored=True,config_restored=True,no_deployment=True))
                    save(destination / 'progress.json',dict(status='complete',inferred=inferred)); return
                totals = {v:{k:sum(r[v][k] for r in rows) for k in ('tp','unmatched','fn','predictions','targets')} for v in ('current','trial')}
                assert totals['current'] == load(MEDIAN / stage / 'report.json')['summary']['trial']
                normal = sum(r['trial']['predictions'] for r in rows if r['image'].startswith('normal_'))
                gain = totals['trial']['tp'] > totals['current']['tp'] if stage!='outer' else totals['trial']['tp'] >= totals['current']['tp']
                qualifies = gain and totals['trial']['unmatched'] <= totals['current']['unmatched'] and not any(r['lost'] for r in rows) and normal == 0
                save(folder / 'report.json',dict(status='complete',qualifies=qualifies,summary=totals,cases=rows,inferred=inferred,normal_cues=normal))
                stages[stage] = dict(qualifies=qualifies,summary=totals); print(str(dict(stage=stage,**stages[stage])),flush=True)
                if not qualifies:
                    save(destination / 'report.json',dict(status='rejected',stage=stage,stages=stages,pins=pins,seconds=round(time.monotonic()-started,2),no_deployment=True)); save(destination / 'progress.json',dict(status='rejected',stage=stage)); return
        save(destination / 'report.json',dict(status='source_pass_requires_reference_ROI_SAM_Qt',stages=stages,pins=pins,seconds=round(time.monotonic()-started,2),no_deployment=True,field_accuracy=False))
        save(destination / 'progress.json',dict(status='complete',stages=stages))
    except BaseException as error:
        save(destination / 'progress.json',dict(status='failed',error=type(error).__name__+': '+str(error))); raise


if __name__ == '__main__': main()
