"""Conditional TRAIN192 common scale. Requires an independently audited rejection.

NOT launched; a frozen whole-source auditor is mandatory before launch.
"""
import copy
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
sys.dont_write_bytecode = True
from prepare_paired_port_semantics import ROOT, REPO, DATA, load, save, sha, read_image, REFERENCE_SHA
from current_port_baseline_audit import read_targets
from run_allport480_source import NEW_SHA, TRAIN
from inspection_agent.paired_native_pose import native_pose_runtime_fingerprint, HEAD_SHA, HEAD_RELATIVE
from inspection_agent.paired_port_features import expected_in_source, valid_boxes, embeddings, paired_features
from inspection_agent.optional_port_crop_review import CONFIG
from inspection_agent.teacher_student_port_support import STUDENT_SHA, STUDENT_RELATIVE
from inspection_agent.feature_residual_port_support import WEIGHT_SHA, WEIGHT_RELATIVE
from raw_pose_consensus import proposals
from fine_voter_quality import attach
from consensus_rank import select
from audit_port_multiscale_acceptance import matches, metric
from exact_paired_score_cache import binding, reuse
from paired_common_scale640 import infer
from common_scale_evidence import (audit_common_views, validate_fallback_prerequisites,
                                  check_common_prefixes, common_source_qualifies)

PARENT = ROOT/'artifacts/allport480_teacher_source_20261005'
PARENT_AUDIT = ROOT/'artifacts/allport480_teacher_source_audit_20261005/report.json'
DEV = ROOT/'artifacts/allport480_teacher_holdouts_20261005'
DEV_AUDIT = ROOT/'artifacts/allport480_teacher_holdout_audit_20261005/report.json'
OUT = ROOT/'artifacts/allport_common640_source_20261005'
PLAN = ROOT/'artifacts/allport_common640_preregistration_20261005/PLAN.md'
AUDITOR = Path(__file__).with_name('audit_allport_common640_source.py')


def main():
    if OUT.exists(): raise FileExistsError('preserve every source attempt; never resume/overwrite')
    # Mandatory auditor must exist BEFORE outputs/model imports. No live rerun.
    if not AUDITOR.is_file(): raise ValueError('prepare and freeze the independent whole-source auditor first')
    parent = load(PARENT/'report.json'); parent_audit = load(PARENT_AUDIT)
    dev = load(DEV/'report.json'); dev_audit = load(DEV_AUDIT)
    validate_fallback_prerequisites(parent, parent_audit, dev, dev_audit)
    for directory, result, audit in [(PARENT, parent, parent_audit), (DEV, dev, dev_audit)]:
        if (audit['source_report_sha256'] != sha(directory/'report.json')
                or audit['source_protocol_sha256'] != sha(directory/'protocol.json')
                or any(sha(p) != d for p, d in result['pins'].items())
                or any(sha(p) != d for p, d in audit['source_case_sha256'].items())):
            raise ValueError('completed independently audited prerequisite drift')
    if len(parent_audit['source_case_sha256']) != 192: raise ValueError('exact completed source inventory required')
    frozen = native_pose_runtime_fingerprint(REPO)
    protocol = load(PARENT/'protocol.json')
    if frozen != parent['runtime'] or frozen != dev['runtime'] or CONFIG != protocol['original_config']:
        raise ValueError('protect live mainline/runtime/options')
    roles = dict(teacher=NEW_SHA, student=STUDENT_SHA, feature=WEIGHT_SHA)
    if roles != protocol['roles']: raise ValueError('fixed checkpoint roles changed')
    paths = dict(teacher=Path(load(TRAIN/'full/report.json')['last_checkpoint']),
                 student=REPO/STUDENT_RELATIVE, feature=REPO/WEIGHT_RELATIVE)
    if any(sha(p) != roles[k] for k, p in paths.items()): raise ValueError('checkpoint identity drift')
    referencepath = DATA/'images/train01/normal_073.JPG'
    if sha(referencepath) != REFERENCE_SHA or sha(REPO/HEAD_RELATIVE) != HEAD_SHA:
        raise ValueError('fixed reference/head drift')
    dirty = lambda: subprocess.check_output(['E:/Git/cmd/git.exe', '-C', str(REPO), 'status', '--porcelain'], text=True)
    before_dirty = dirty(); original_config = copy.deepcopy(CONFIG)
    if before_dirty != protocol['mainline_dirty_before']: raise ValueError('preserve all existing dirty worktree changes')
    entries = parent['cases']; names = [r['image'] for r in entries]
    if len(set(names)) != 192 or sorted(names) != sorted(protocol['train_sources']): raise ValueError('TRAIN192 membership drift')
    files = [Path(__file__), AUDITOR, PLAN, PARENT/'report.json', PARENT/'protocol.json', PARENT_AUDIT,
             DEV/'report.json', DEV/'protocol.json', DEV_AUDIT, TRAIN/'full/report.json', referencepath, REPO/HEAD_RELATIVE, *paths.values()]
    files += [Path(__file__).with_name(n+'.py') for n in ['common_scale_evidence', 'paired_common_scale640',
              'raw_pose_consensus', 'fine_voter_quality', 'consensus_rank', 'replacement_voters', 'exact_paired_score_cache',
              'novel_box_geometry', 'relative_port_box', 'prepare_paired_port_semantics', 'current_port_baseline_audit',
              'audit_port_multiscale_acceptance', 'audit_allport480_source']]
    pins = {str(p): sha(p) for p in files}
    import torch, cv2, psutil
    if psutil.virtual_memory().available < 6*2**30: raise RuntimeError('do not compete for active inference memory')
    torch.set_num_threads(2); cv2.setNumThreads(1)
    OUT.mkdir(); (OUT/'config/Ultralytics').mkdir(parents=True)
    shutil.copy2('C:/Windows/Fonts/arial.ttf', OUT/'config/Ultralytics/Arial.ttf')
    os.environ.update(YOLO_CONFIG_DIR=str(OUT/'config'), YOLO_OFFLINE='True', YOLO_AUTOINSTALL='False', HF_HUB_OFFLINE='1')
    from ultralytics import YOLO
    class Capped:
        def __init__(self, model): self.model = model
        def predict(self, *a, **kw):
            torch.set_num_threads(2); result = self.model.predict(*a, **kw); torch.set_num_threads(2); return result
    models = {}
    for role, path in paths.items():
        model = YOLO(str(path))
        if model.task != 'segment' or dict(model.names) != {0: 'unplugged_plug', 1: 'unplugged_jack'}:
            raise ValueError('detector class contract changed')
        models[role] = Capped(model)
    head = torch.nn.Linear(6144, 3)
    head.load_state_dict(torch.load(REPO/HEAD_RELATIVE, map_location='cpu', weights_only=True)['state_dict'])
    head.eval().requires_grad_(False)
    reference = read_image(referencepath); encoder = None; rows = []; started = time.monotonic()
    fresh_views = reused_scores = fresh_scores = 0
    (OUT/'train').mkdir()
    save(OUT/'protocol.json', dict(pins=pins, runtime=frozen, roles=roles, original_config=original_config,
         train_sources=names, mainline_dirty_before=before_dirty, wall_bound_seconds=14400,
         same_weight_one_vote=True, GT_after_predictions=True, exact_semantic_cache_only=True,
         no_heldout_inference_or_label_decoding=True, development_integrity_sha_reads_only=True,
         no_deployment=True, protected_prefixes=[295, 298, 300],
         source_gate=dict(tp_strictly_above=300, unmatched_max=4, targets=344, normal_cues=0)))
    try:
        for index, entry in enumerate(entries):
            if time.monotonic()-started > 14400: raise TimeoutError('partial collection is not source acceptance')
            name = entry['image']; stem = Path(name).stem
            def progress(phase):
                save(OUT/'progress.json', dict(status='running', pid=os.getpid(), completed=index, total=192,
                     image=name, phase=phase, fresh_view_calls=fresh_views, fresh_semantic_scores=fresh_scores,
                     reused_semantic_scores=reused_scores, seconds=round(time.monotonic()-started, 2)))
            progress('bound_audited_source')
            p = PARENT/'train'/(stem+'_predictions.json'); pins[str(p)] = sha(p); case = load(p)
            if pins[str(p)] != parent_audit['source_case_sha256'][str(p)]: raise ValueError('parent case drift')
            sourcepath = DATA/'images/train01'/name; pins[str(sourcepath)] = sha(sourcepath)
            if pins[str(sourcepath)] != case['source_sha256']: raise ValueError('source pixels drift')
            original, research, current = case['original'], case['current'], case['trial']
            check_common_prefixes(original, research, current, current)
            alignment = case['alignment']; eligible = case['eligible']; reason = case['skip_reason']
            if len(current['all_predictions'])-len(current['primary']) >= 5: eligible = False; reason = 'shared_budget_full'
            views = []; native = []; scores = []; score_reuse = None
            if eligible:
                image = read_image(sourcepath)
                if image.shape[:2] != (2736, 3648): raise ValueError('source frame drift')
                views = copy.deepcopy(case['new_voter_views'])
                for role in roles:
                    progress(role+'_fresh_common640stride480')
                    pred = infer(models[role], image); fresh_views += 1
                    views.append(dict(source_sha256=pins[str(sourcepath)], weight_sha256=roles[role],
                                      view='fresh_common640stride480', predictions=pred))
                audit_common_views(views, roles, pins[str(sourcepath)], image.shape[:2])
                expected, valid = expected_in_source(reference, alignment['source_to_reference_homography'], image.shape[:2])
                frame = dict(source_sha256=pins[str(sourcepath)], predictions={'source_shape': list(image.shape[:2])})
                native = proposals(frame, views, current)
                native = [native[i] for i in valid_boxes([r['box_xyxy'] for r in native], valid)]
                native = attach(native, views, pins[str(sourcepath)], image.shape[:2])
                if native:
                    old = binding(case['source_sha256'], REFERENCE_SHA, parent['runtime'], alignment, image.shape[:2])
                    new = binding(pins[str(sourcepath)], REFERENCE_SHA, frozen, alignment, image.shape[:2])
                    scores, missing, count = reuse(native, case['proposals'], case['probabilities'], old, new)
                    score_reuse = dict(source_case_path=str(p), source_case_sha256=pins[str(p)], old_input_binding=old,
                         new_input_binding=new, reused_count=count, fresh_indices=missing, same_classifier_not_an_extra_vote=True)
                    reused_scores += count
                    if missing:
                        progress('fresh_missing_paired_semantics')
                        if encoder is None:
                            import dino_feature_diff as dino
                            encoder = dino._model(); encoder.eval().requires_grad_(False); torch.set_num_threads(2)
                        boxes = [native[i]['box_xyxy'] for i in missing]
                        vectors = paired_features(embeddings(encoder, image, boxes), embeddings(encoder, expected, boxes))
                        with torch.inference_mode(): values = head(vectors).softmax(1).tolist()
                        for i, value in zip(missing, values): scores[i] = value
                        fresh_scores += len(missing)
            trial = select(current, native, scores, HEAD_SHA)
            check_common_prefixes(original, research, current, trial)
            save(OUT/'train'/(stem+'_predictions.json'), dict(image=name, original=original, research=research, current=current,
                 trial=trial, alignment=alignment, eligible=eligible, skip_reason=reason, source_sha256=pins[str(sourcepath)],
                 head_sha256=HEAD_SHA, roles=roles, new_voter_views=views, proposals=native, probabilities=scores, semantic_score_reuse=score_reuse))
            # Predictions already persisted; only now decode original source GT.
            targets = read_targets('train', name, [2736, 3648], parent['pins'][str(DATA/'labels/train01'/(stem+'.txt'))], pins)
            versions = dict(original=original, research=research, current=current, trial=trial)
            hits = {k: matches(v['all_predictions'], targets)[0] for k, v in versions.items()}
            rows.append(dict(image=name, **{k: metric(v['all_predictions'], targets) for k, v in versions.items()},
                 gained=sorted(hits['trial']-hits['current']), lost=sorted(hits['current']-hits['trial']),
                 lost_original=sorted(hits['original']-hits['trial']), lost_research=sorted(hits['research']-hits['trial']), skip_reason=reason))
            save(OUT/'partial_metrics.json', dict(status='incomplete_not_acceptance', completed=index+1, total=192, cases=rows))
            print(dict(completed=index+1, total=192, image=name), flush=True)
        totals = {v: {k: sum(r[v][k] for r in rows) for k in ('tp', 'unmatched', 'fn', 'predictions', 'targets')}
                  for v in ('original', 'research', 'current', 'trial')}
        normal = sum(r['trial']['predictions'] for r in rows if r['image'].startswith('normal_'))
        qualifies = common_source_qualifies(totals, normal, rows)
        if (any(sha(p) != d for p, d in pins.items()) or any(sha(p) != d for p, d in parent['pins'].items())
                or native_pose_runtime_fingerprint(REPO) != frozen or dirty() != before_dirty or CONFIG != original_config):
            raise ValueError('input/runtime/mainline drift')
        result = dict(status='source_pass_requires_independent_replay_and_fresh_holdouts' if qualifies else 'rejected',
             qualifies=qualifies, summary=totals, cases=rows, normal_cues=normal, pins=pins, runtime=frozen,
             fresh_view_calls=fresh_views, reused_semantic_scores=reused_scores, fresh_semantic_scores=fresh_scores,
             seconds=round(time.monotonic()-started, 2), mainline_unchanged=True, no_deployment=True, field_accuracy=None)
        save(OUT/'report.json', result); save(OUT/'progress.json', {k: result[k] for k in ('status', 'seconds', 'fresh_view_calls')})
        print(dict(status=result['status'], summary=totals), flush=True)
    except BaseException as exc:
        save(OUT/'progress.json', dict(status='failed', completed=len(rows), total=192,
             error=type(exc).__name__+': '+str(exc), partial_is_not_acceptance=True)); raise


if __name__ == '__main__': main()
