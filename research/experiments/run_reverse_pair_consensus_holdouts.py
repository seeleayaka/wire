"""Pinned fine INNER reuse with new original-pixel scores; conditional fresh OUTER."""
import os
import sys
import time
import shutil
from pathlib import Path
sys.dont_write_bytecode = True
from prepare_paired_port_semantics import ROOT, REPO, DATA, load, save, sha, read_image, REFERENCE_SHA
from current_port_baseline_audit import BASE, read_current_case, read_targets
from inspection_agent.paired_native_pose import native_pose_runtime_fingerprint, HEAD_RELATIVE, HEAD_SHA
from inspection_agent.paired_port_features import expected_in_source, valid_boxes, embeddings, paired_features
from inspection_agent.teacher_student_port_support import TEACHER_RELATIVE, TEACHER_SHA, STUDENT_RELATIVE, STUDENT_SHA
from raw_pose_consensus import proposals
from fine_voter_quality import attach
from consensus_rank import select
from audit_port_multiscale_acceptance import metric, matches

SOURCE = ROOT / 'artifacts/reverse_pair_consensus_20261004'
AUDIT = ROOT / 'artifacts/reverse_pair_consensus_replay_20261004/report.json'
PREVIOUS = ROOT / 'artifacts/fine_consensus_rank_holdouts_20261004'
CURRENT = ROOT / 'artifacts/paired_pose_native_three_20261004'
PREP = ROOT / 'artifacts/paired_support_graph_20261003/source_selections'
OUT = ROOT / 'artifacts/reverse_pair_consensus_holdouts_20261004'
PLAN = ROOT / 'artifacts/reverse_pair_consensus_holdouts_preregistration_20261004/PLAN.md'


def main():
    import cv2
    import torch
    import psutil
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    if OUT.exists():
        raise FileExistsError('Preserve combined held-out trial')
    assert psutil.virtual_memory().available > 6*2**30
    source = load(SOURCE / 'report.json')
    assert source['status'] == 'source_pass_requires_fresh_holdouts_and_actual_gates'
    audit = load(AUDIT)
    assert audit['status'] == 'pass' and audit['source_report_sha256'] == sha(SOURCE / 'report.json')
    previous = load(PREVIOUS / 'report.json')
    assert previous['status'] == 'rejected' and previous['failed_stage'] == 'inner'
    runtime = native_pose_runtime_fingerprint(REPO)
    assert runtime == source['runtime'] == previous['runtime']
    for report in (source, previous):
        assert all(sha(Path(p)) == v for p, v in report['pins'].items())
    referencepath = DATA / 'images/train01/normal_073.JPG'
    assert sha(referencepath) == REFERENCE_SHA
    headpath = SOURCE / 'full/last_head.pt'
    hsha = sha(headpath)
    assert source['pins'][str(headpath)] == hsha
    paths = [Path(__file__), PLAN, SOURCE / 'report.json', AUDIT, PREVIOUS / 'report.json',
             PREVIOUS / 'protocol.json', headpath, REPO / HEAD_RELATIVE, referencepath]
    paths += [Path(__file__).with_name(n) for n in ('raw_pose_consensus.py', 'fine_voter_quality.py', 'consensus_rank.py', 'novel_box_geometry.py', 'paired_fine_tile_views.py')]
    pins = {str(p): sha(p) for p in paths}
    for p, digest in previous['pins'].items():
        pins[p] = digest
    state = torch.load(headpath, map_location='cpu', weights_only=True)
    assert state['classes'] == ['other', 'unplugged_plug', 'unplugged_jack'] and state['encoder_sha256'] == runtime['encoder']
    head = torch.nn.Linear(6144, 3)
    head.load_state_dict(state['state_dict'])
    head.eval().requires_grad_(False)
    oldhead = torch.nn.Linear(6144, 3)
    oldhead.load_state_dict(torch.load(REPO / HEAD_RELATIVE, map_location='cpu', weights_only=True)['state_dict'])
    oldhead.eval().requires_grad_(False)
    assert sha(REPO / HEAD_RELATIVE) == HEAD_SHA
    OUT.mkdir()
    reference, encoder = read_image(referencepath), None
    start, stages = time.monotonic(), {}
    save(OUT / 'protocol.json', dict(pins=pins, runtime=runtime, head_sha256=hsha, INNER_detector_reuse=True,
        source_report_sha256=sha(SOURCE / 'report.json'), original_pixel_features_reextracted=True,
        probability_gate=.98, distinct_SHA_votes=3, shared_budget='5+5', no_training=True,
        no_deployment=True, field_accuracy=False))
    def progress(**kw):
        save(OUT / 'progress.json', dict(status='running', pid=os.getpid(), seconds=round(time.monotonic()-start, 2), **kw))
    def finish(failed):
        assert all(sha(Path(p)) == v for p, v in pins.items()) and native_pose_runtime_fingerprint(REPO) == runtime
        result = dict(status='rejected' if failed else 'holdout_pass_requires_actual_reference_ROI_and_Qt_SAM',
            failed_stage=failed, stages=stages, runtime=runtime, pins=pins,
            INNER_detector_reuse=True, original_pixel_features_reextracted=True, no_training=True,
            no_deployment=True, field_accuracy=False, seconds=round(time.monotonic()-start, 2))
        save(OUT / 'report.json', result)
        save(OUT / 'progress.json', dict(status=result['status'], seconds=result['seconds']))
        print({k: v for k, v in result.items() if k not in ('pins', 'runtime')}, flush=True)
    try:
        for stage, count, base_tp, base_fp in [('inner', 48, 68, 0), ('outer', 30, 40, 1)]:
            folder = OUT / stage
            folder.mkdir()
            indexpath = PREP / stage / 'index.json'
            pins[str(indexpath)] = sha(indexpath)
            indexed = {r['image']: r for r in load(indexpath)['records']}
            entries = load(BASE / stage / 'report.json')['cases']
            assert len(entries) == count
            rows, reused, fresh, vectors_count = [], 0, 0, 0
            if stage == 'outer':
                configdir = OUT / 'config'
                (configdir / 'Ultralytics').mkdir(parents=True)
                shutil.copy2('C:/Windows/Fonts/arial.ttf', configdir / 'Ultralytics/Arial.ttf')
                os.environ.update(YOLO_CONFIG_DIR=str(configdir), YOLO_OFFLINE='True', YOLO_AUTOINSTALL='False', HF_HUB_OFFLINE='1')
                from ultralytics import YOLO
                from paired_fine_tile_views import infer
                import assembly_auto_review_robust_v3 as registration
                from paired_port_semantics import CachedReferenceSIFT
                weights = [REPO / TEACHER_RELATIVE, REPO / STUDENT_RELATIVE]
                digests = [TEACHER_SHA, STUDENT_SHA]
                assert [sha(p) for p in weights] == digests
                pins.update({str(p): d for p, d in zip(weights, digests)})
                class Capped:
                    def __init__(self, model):
                        self.model = model
                    def predict(self, *args, **kwargs):
                        torch.set_num_threads(2)
                        result = self.model.predict(*args, **kwargs)
                        torch.set_num_threads(2)
                        return result
                models = [Capped(YOLO(str(p))) for p in weights]
            for completed, entry in enumerate(entries):
                name = entry['image']
                progress(stage=stage, image=name, completed=completed, total=count, phase='original_pixel_pairs_new_head', fresh_feature_candidates=vectors_count)
                sourcepath = DATA / 'images' / ('val01' if stage == 'outer' else 'train01') / name
                source_sha = sha(sourcepath)
                pins[str(sourcepath)] = source_sha
                pairpath = Path(indexed[name]['path'])
                assert sha(pairpath) == indexed[name]['sha256']
                pins[str(pairpath)] = sha(pairpath)
                pair = load(pairpath)
                teacher, old = read_current_case(stage, entry, pins)
                assert teacher == pair['teacher']
                pool = [teacher, pair['student'], pair['feature'], old['alternative']]
                currentpath = CURRENT / stage / (Path(name).stem+'_predictions.json')
                pins[str(currentpath)] = sha(currentpath)
                accepted = load(currentpath)
                assert accepted['head_sha256'] == HEAD_SHA
                current = accepted['trial']
                remaining = 5 - (len(current['all_predictions'])-len(current['primary']))
                eligible = remaining > 0 and any(p['confidence'] > .05 for m in pool for p in m['predictions']['merged_predictions'])
                alignment, views, native, values = None, [], [], []
                reason, featurefile, old_consistency = None, None, None
                old_case = None
                if stage == 'inner':
                    oldpath = PREVIOUS / stage / currentpath.name
                    pins[str(oldpath)] = sha(oldpath)
                    old_case = load(oldpath)
                    assert old_case['source_sha256'] == source_sha and old_case['current'] == current and old_case['eligible'] == eligible
                    assert old_case['head_sha256'] == HEAD_SHA
                    alignment, views, reason = old_case['alignment'], old_case['new_views'], old_case['skip_reason']
                    for model in views:
                        assert model['weight_sha256'] in (TEACHER_SHA, STUDENT_SHA) and model['source_sha256'] == source_sha
                        assert model['view'] == 'fresh_fine960_stride720'
                    reused += bool(views)
                elif eligible:
                    image = read_image(sourcepath)
                    with CachedReferenceSIFT(reference):
                        cv2.setRNGSeed(0)
                        _, alignment = registration.automatic_homography(reference, image)
                    if alignment.get('alignment_quality', {}).get('reliable'):
                        for model, digest in zip(models, digests):
                            views.append(dict(image=name, source_sha256=source_sha, weight_sha256=digest,
                                predictions=infer(model, image), view='fresh_fine960_stride720'))
                        fresh += 1
                    else:
                        reason = 'unreliable_original_SIFT'
                else:
                    reason = 'budget_full_or_no_existing_seed_evidence'
                if views:
                    image = read_image(sourcepath)
                    assert image.shape[:2] == (2736, 3648) and alignment['alignment_quality']['reliable']
                    expected, mask = expected_in_source(reference, alignment['source_to_reference_homography'], image.shape[:2])
                    native = proposals(teacher, pool+views, current)
                    native = [native[i] for i in valid_boxes([p['box_xyxy'] for p in native], mask)]
                    native = attach(native, pool+views, source_sha, image.shape[:2])
                    if old_case is not None:
                        assert native == old_case['proposals']
                    if native:
                        import dino_feature_diff as dino
                        if encoder is None:
                            encoder = dino._model()
                            encoder.eval().requires_grad_(False)
                            torch.set_num_threads(2)
                        boxes = [p['box_xyxy'] for p in native]
                        vectors = paired_features(embeddings(encoder, image, boxes), embeddings(encoder, expected, boxes))
                        vectors_count += len(vectors)
                        with torch.inference_mode():
                            if old_case is not None:
                                reproduced = oldhead(vectors).softmax(1)
                                old_probs = torch.tensor(old_case['probabilities']).reshape(len(native), 3)
                                torch.testing.assert_close(reproduced, old_probs, atol=1e-6, rtol=1e-5)
                                old_consistency = float((reproduced-old_probs).abs().max())
                            values = head(vectors).softmax(1).tolist()
                        featurepath = folder / (Path(name).stem+'_features.pt')
                        torch.save(dict(features=vectors, source_sha256=source_sha, boxes=boxes), featurepath)
                        pins[str(featurepath)] = sha(featurepath)
                        featurefile = str(featurepath)
                if old_case is not None:
                    assert native == old_case['proposals']
                trial = select(current, native, values, hsha)
                save(folder / currentpath.name, dict(image=name, current=current, trial=trial, proposals=native,
                    probabilities=values, head_sha256=hsha, new_views=views, alignment=alignment,
                    source_sha256=source_sha, detector_reused=stage == 'inner', eligible=eligible,
                    skip_reason=reason, feature_file=featurefile, original_head_consistency_max_delta=old_consistency))
                gt = read_targets(stage, name, [2736, 3648], entry['label_sha256'], pins)
                a, b = matches(current['all_predictions'], gt)[0], matches(trial['all_predictions'], gt)[0]
                rows.append(dict(image=name, current=metric(current['all_predictions'], gt), trial=metric(trial['all_predictions'], gt),
                                 gained=sorted(b-a), lost=sorted(a-b), skip_reason=reason))
            totals = {v: {k: sum(r[v][k] for r in rows) for k in ('tp', 'unmatched', 'fn', 'predictions', 'targets')} for v in ('current', 'trial')}
            assert totals['current']['tp'] == base_tp and totals['current']['unmatched'] == base_fp
            normal = sum(r['trial']['predictions'] for r in rows if r['image'].startswith('normal_'))
            qualifies = (totals['trial']['tp'] > base_tp if stage == 'inner' else totals['trial']['tp'] >= base_tp) and totals['trial']['unmatched'] <= base_fp and normal == 0 and not any(r['lost'] for r in rows)
            summary = dict(qualifies=qualifies, summary=totals, normal_cues=normal, reused_fine_sources=reused,
                           fresh_fine_sources=fresh, fresh_feature_candidates=vectors_count)
            stages[stage] = summary
            save(folder / 'report.json', dict(status='complete', **summary, cases=rows))
            print(dict(stage=stage, **summary), flush=True)
            if not qualifies:
                finish(stage)
                return
        finish(None)
    except BaseException as error:
        save(OUT / 'progress.json', dict(status='failed', error=type(error).__name__+': '+str(error)))
        raise


if __name__ == '__main__':
    main()
