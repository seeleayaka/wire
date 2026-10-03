"""Training-first append-only trial; pinned replays plus necessary fresh inference."""
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
REPO = Path('E:/PythonProject10')
sys.path.insert(0, str(REPO))
OUT = ROOT / 'artifacts/port_residual_feature_support_20261003'
FULL = ROOT / 'artifacts/port_feature_adaptation_20261003'
PAIR = ROOT / 'artifacts/teacher_student_port_20261003'
TEACHER = ROOT / 'artifacts/core_port_recheck_20261002'
STUDENT = ROOT / 'artifacts/port_training_multiscale_20261002'
EXTENDED = ROOT / 'artifacts/port_extended_training_controls_20261003'
DATA = REPO / 'data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults'
from inspection_agent.optional_port_crop_review import sha, read_image, predict
from inspection_agent.context_port_recheck import predict_seed_views, recheck_proposals
from inspection_agent.teacher_student_port_support import native_selection, TEACHER_SHA, STUDENT_SHA
from port_residual_feature_support import merge_residual, POLICY_ID
from teacher_student_port_policy import merge
from audit_port_multiscale_acceptance import matches, metric


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def qualifies(mode, current, trial, normal, losses):
    gain = trial['tp'] > current['tp'] if mode in ('train', 'inner') else trial['tp'] >= current['tp']
    return gain and trial['unmatched'] <= current['unmatched'] and normal == 0 and losses == 0


def main():
    if OUT.exists():
        raise FileExistsError('Preserve all earlier trials; fresh output required')
    full = load(FULL / 'full/report.json')
    assert full['status'] == 'complete' and full['completed_epochs'] == 2
    assert full['changed_new_backbone_tensors'] > 0 and full['frozen_parameters_changed'] == 0
    weight = Path(full['weights']['last.pt']['path'])
    weight_sha = sha(weight)
    assert weight_sha == full['weights']['last.pt']['sha256']
    groups = load(STUDENT / 'protocol.json')
    assert set(groups['train_sources']).isdisjoint(groups['inner_val_sources'])
    pins = {str(p): sha(p) for p in (
        Path(__file__), Path(__file__).with_name('port_residual_feature_support.py'),
        Path(__file__).with_name('teacher_student_port_policy.py'),
        Path(__file__).with_name('core_port_recheck_policy.py'),
        Path(__file__).with_name('core_port_supplement_policy.py'),
        Path(__file__).with_name('core_port_precision_policy.py'),
        FULL / 'full/report.json', FULL / 'protocol.json', weight,
        STUDENT / 'protocol.json', EXTENDED / 'protocol.json', EXTENDED / 'report.json',
        REPO / 'inspection_agent/context_port_recheck.py',
        REPO / 'inspection_agent/teacher_student_port_support.py',
        REPO / 'inspection_agent/optional_port_crop_review.py')}
    # Pin every existing prediction and source before *any* trial scoring.
    cases = {}
    for mode in ('train', 'extended', 'inner', 'outer'):
        if mode == 'extended':
            names = load(EXTENDED / 'protocol.json')['images']
        else:
            names = [p['image'] for p in load(PAIR / mode / 'report.json')['cases']]
        assert len(names) == dict(train=32, extended=136, inner=48, outer=30)[mode]
        if mode in ('train', 'extended'):
            assert set(names) <= set(groups['train_sources'])
        if mode == 'inner':
            assert set(names) == set(groups['inner_val_sources'])
        cases[mode] = []
        for name in names:
            stem = Path(name).stem
            paths = ([EXTENDED / (stem + '_predictions.json')] if mode == 'extended' else
                     [TEACHER / mode / (stem + '_zoom_predictions.json'),
                      STUDENT / 'evaluation' / mode / (stem + '_zoom_predictions.json')])
            feature_path = FULL / 'evaluation' / mode / (stem + '_zoom_predictions.json')
            if mode in ('train', 'inner'):
                paths.append(feature_path)
            for path in paths:
                pins[str(path)] = sha(path)
            if mode == 'extended':
                record = load(paths[0]); teacher = record['versions']['teacher']; student = record['versions']['student']
            else:
                teacher, student = load(paths[0]), load(paths[1])
            source = DATA / 'images' / ('val01' if mode == 'outer' else 'train01') / name
            pins[str(source)] = sha(source)
            assert teacher['source_sha256'] == student['source_sha256'] == pins[str(source)]
            assert teacher['weight_sha256'] == TEACHER_SHA and student['weight_sha256'] == STUDENT_SHA
            feature = load(feature_path) if mode in ('train', 'inner') else None
            if feature is not None:
                assert feature['source_sha256'] == pins[str(source)] and feature['weight_sha256'] == weight_sha
            cases[mode].append(dict(image=name, source=source, teacher=teacher, student=student, feature=feature))
    (OUT / 'config/Ultralytics').mkdir(parents=True)
    shutil.copy2('C:/Windows/Fonts/arial.ttf', OUT / 'config/Ultralytics/Arial.ttf')
    save(OUT / 'protocol.json', dict(policy_id=POLICY_ID, pins=pins, stages=['train', 'extended', 'inner', 'outer'],
        require_train_inner_gain=True, preserve_every_accepted_cue=True, maximum_primary=5, maximum_extra=5,
        thresholds_unchanged=True, prediction_selection_before_labels=True,
        negative_selection='ALL136 previously unused training normal/damaged/misrouted sources',
        replay=['train32', 'inner48'], fresh_feature_inference=['eligible extended', 'eligible outer'],
        exact_short_circuit='No same-class teacher raw confidence>.25 means no residual can pass; no feature inference needed',
        old_validation_already_seen=True, field_accuracy=False, automatic_deployment=False))
    os.environ.update(YOLO_OFFLINE='True', YOLO_AUTOINSTALL='False', HF_HUB_OFFLINE='1',
                      YOLO_CONFIG_DIR=str(OUT / 'config'), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    capped = None
    started = time.monotonic()
    try:
        for mode in ('train', 'extended', 'inner', 'outer'):
            directory = OUT / mode; directory.mkdir()
            inferred = 0; short_circuited = 0; selected = []
            for index, case in enumerate(cases[mode], 1):
                teacher, student = case['teacher'], case['student']
                accepted = merge(teacher, student)
                assert accepted['fallback_reason'] is None
                feature = case['feature']
                if feature is None:
                    raw_support = any(p['confidence'] > .25 for p in teacher['predictions']['merged_predictions'])
                    has_slot = len(accepted['all_predictions']) - len(accepted['primary']) < 5
                    if raw_support and has_slot:
                        if capped is None:
                            import torch
                            from ultralytics import YOLO
                            torch.set_num_threads(4); model = YOLO(str(weight))
                            assert model.task == 'segment' and dict(model.names) == {0: 'unplugged_plug', 1: 'unplugged_jack'}
                            class Capped:
                                def predict(self, *a, **kw):
                                    result = model.predict(*a, **kw); torch.set_num_threads(4); return result
                            capped = Capped()
                        image = read_image(case['source']); raw = predict(capped, image)
                        feature = dict(image=case['image'], source_sha256=teacher['source_sha256'], weight_sha256=weight_sha,
                                       predictions=raw, zoom_evidence=[])
                        if len(native_selection(feature)['supplementary']) < 5:
                            feature['zoom_evidence'] = predict_seed_views(capped, image, recheck_proposals(raw))
                        inferred += 1
                    else:
                        feature = dict(image=case['image'], source_sha256=teacher['source_sha256'], weight_sha256=weight_sha,
                            predictions=dict(source_shape=teacher['predictions']['source_shape'], merged_predictions=[], edge_kept_predictions=[]),
                            zoom_evidence=[])
                        short_circuited += 1
                trial = merge_residual(teacher, student, feature)
                assert trial['feature_fallback_reason'] is None
                assert trial['all_predictions'][:len(accepted['all_predictions'])] == accepted['all_predictions']
                record = dict(image=case['image'], accepted=accepted, trial=trial, feature=feature)
                save(directory / (Path(case['image']).stem + '_predictions.json'), record)
                selected.append(record)
                save(OUT / 'progress.json', dict(status='running', mode=mode, completed=index, total=len(cases[mode]),
                     fresh_feature_inferred=inferred, exact_short_circuits=short_circuited, elapsed_seconds=round(time.monotonic() - started, 2)))
                print(f'{mode} {index}/{len(cases[mode])} {case["image"]} additions={len(trial["feature_additions"])}', flush=True)
            totals = {k: dict(tp=0, unmatched=0, fn=0, predictions=0, targets=0) for k in ('current', 'trial')}
            lost = 0; normal = 0; entries = []
            for record in selected:
                name = record['image']; label = DATA / 'labels' / ('val01' if mode == 'outer' else 'train01') / (Path(name).stem + '.txt')
                h, w = record['feature']['predictions']['source_shape']; targets = []
                for line in label.read_text(encoding='utf-8').splitlines():
                    cls, cx, cy, bw, bh = map(float, line.split())
                    if cls in (3, 4):
                        targets.append(dict(class_id=int(cls)-3, box=[(cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h]))
                old, new = record['accepted']['all_predictions'], record['trial']['all_predictions']
                metrics = dict(current=metric(old, targets), trial=metric(new, targets))
                old_hits, new_hits = matches(old, targets)[0], matches(new, targets)[0]
                lost += len(old_hits-new_hits)
                if name.startswith('normal_'): normal += len(new)
                for key, values in metrics.items():
                    for field, value in values.items(): totals[key][field] += value
                entries.append(dict(image=name, metrics=metrics, lost_targets=sorted(old_hits-new_hits),
                    gained_targets=sorted(new_hits-old_hits), feature_additions=len(record['trial']['feature_additions']), label_sha256=sha(label)))
            baseline = (load(EXTENDED / 'report.json')['summary']['current'] if mode == 'extended' else
                        load(PAIR / mode / 'report.json')['summary']['cross_model_supported'])
            assert totals['current'] == baseline
            assert {p: sha(Path(p)) for p in pins} == pins
            accepted = qualifies(mode, totals['current'], totals['trial'], normal, lost)
            report = dict(status='complete', summary=totals, qualifies=accepted, lost_targets=lost, normal_cues=normal,
                cases=entries, fresh_feature_inferred=inferred, exact_short_circuits=short_circuited,
                prediction_replay=mode in ('train','inner'), automatic_deployment=False, field_accuracy=False,
                reference_gates_not_evaluated=True, elapsed_seconds=round(time.monotonic()-started, 2))
            save(directory / 'report.json', report)
            print(json.dumps(dict(mode=mode, summary=totals, qualifies=accepted, lost=lost)), flush=True)
            if not accepted:
                save(OUT / 'progress.json', dict(status='rejected', mode=mode, summary=totals, automatic_deployment=False))
                return
        save(OUT / 'progress.json', dict(status='source_only_pass_needs_live_reference_and_gui', automatic_deployment=False,
            elapsed_seconds=round(time.monotonic()-started, 2)))
    except Exception as error:
        save(OUT / 'progress.json', dict(status='failed', error=type(error).__name__ + ': ' + str(error), automatic_deployment=False))
        raise


if __name__ == '__main__':
    main()
