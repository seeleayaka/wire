"""Frozen current-policy audit helpers; score only after source selection.

This is cached, previously seen source localization, not field accuracy.
The source/label hashes and every original selected cue are protected.
"""
import copy
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
REPO = Path('E:/PythonProject10')
sys.path.insert(0, str(REPO))
BASE = ROOT / 'artifacts/resolution_plug_final_budget_20261003'
EXTRA = ROOT / 'artifacts/remaining_training_ports_20261003'
DATA = REPO / 'data/external_datasets/mendeley_electrical_wiring_faults/Predictive Maintenance for Electrical Wiring Faults'
OUT = ROOT / 'artifacts/current_port_baseline_audit_20261003_v2'
from inspection_agent.optional_port_crop_review import sha
from inspection_agent.teacher_student_port_support import native_selection
from inspection_agent.resolution_loose_plug_support import resolution_candidates, resolution_runtime_fingerprint
from inspection_agent.port_tiling import box_iou
from audit_port_multiscale_acceptance import metric, matches


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def teacher_path(stage, name):
    stem = Path(name).stem
    if stage == 'train' and (EXTRA / (stem + '_predictions.json')).exists():
        return EXTRA / (stem + '_predictions.json'), 'teacher'
    extended = ROOT / 'artifacts/port_extended_training_controls_20261003' / (stem + '_predictions.json')
    if stage == 'train' and not name.startswith('disconnected_') and extended.exists():
        return extended, 'versions.teacher'
    return ROOT / 'artifacts/core_port_recheck_20261002' / stage / (stem + '_zoom_predictions.json'), None


def read_current_case(stage, entry, pins):
    name = entry['image']
    recordpath = BASE / stage / (Path(name).stem + '_predictions.json')
    record = load(recordpath)
    path, key = teacher_path(stage, name)
    teacher = load(path)
    if key:
        for part in key.split('.'):
            teacher = teacher[part]
    source = DATA / 'images' / ('val01' if stage == 'outer' else 'train01') / name
    for item in (recordpath, path, source):
        pins[str(item)] = sha(item)
    assert teacher['image'] == name == record['image']
    assert teacher['source_sha256'] == record['alternative']['source_sha256'] == pins[str(source)]
    assert resolution_candidates(teacher, record['current'], record['alternative']) == record['trial']
    return teacher, record


def read_targets(stage, name, shape, expected_sha, pins):
    label = DATA / 'labels' / ('val01' if stage == 'outer' else 'train01') / (Path(name).stem + '.txt')
    digest = sha(label)
    assert digest == expected_sha
    pins[str(label)] = digest
    h, w = shape
    targets = []
    for line in label.read_text(encoding='utf-8').splitlines():
        cls, cx, cy, bw, bh = map(float, line.split())
        if cls in (3, 4):
            targets.append(dict(class_id=int(cls)-3, box=[(cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h]))
    return targets


def versions(teacher, record):
    old = record['current']
    native = copy.deepcopy(old['primary'] + old['supplementary'] + old['zoom'])
    # The historical experiment's recheck metadata schema is not the formal
    # schema. Keep the exact saved rows; separately verify geometric parity.
    geometry = lambda rows: [(r['box_xyxy'], r['class_id'], r['confidence']) for r in rows]
    assert geometry(native) == geometry(native_selection(teacher)['all_predictions'])
    pair = native + copy.deepcopy(old['student_additions'])
    assert pair + old['feature_additions'] == old['all_predictions']
    assert old['all_predictions'] + record['trial']['resolution_additions'] == record['trial']['all_predictions']
    return dict(teacher=native, pair=pair, feature=old['all_predictions'], current=record['trial']['all_predictions'])


def main():
    if OUT.exists():
        raise FileExistsError('Preserve previous evidence')
    frozen = resolution_runtime_fingerprint(REPO)
    pins = {str(Path(__file__)): sha(Path(__file__))}
    OUT.mkdir()
    summary = {}
    for stage, count in (('train', 192), ('inner', 48), ('outer', 30)):
        reportpath = BASE / stage / 'report.json'
        pins[str(reportpath)] = sha(reportpath)
        report = load(reportpath)
        assert report['status'] == 'complete' and report['qualifies']
        assert len(report['cases']) == count
        rows = []
        totals = {v: {k: 0 for k in ('tp', 'unmatched', 'fn', 'predictions', 'targets')} for v in ('teacher', 'pair', 'feature', 'current')}
        causes = dict(no_teacher_support_geometry=0, shared_extra_budget_full=0, eligible_support_but_no_strict_complement=0)
        capacity = dict(global_ten_box_oracle_misses=0, fixed_primary_plus_five_extra_oracle_misses=0)
        for entry in report['cases']:
            teacher, record = read_current_case(stage, entry, pins)
            selected = versions(teacher, record)  # Policy is finalized before labels are opened.
            targets = read_targets(stage, entry['image'], teacher['predictions']['source_shape'], entry['label_sha256'], pins)
            metrics = {key: metric(value, targets) for key, value in selected.items()}
            assert metrics['current'] == entry['metrics']['trial']
            assert metrics['feature'] == entry['metrics']['current']
            for v, m in metrics.items():
                for k, value in m.items():
                    totals[v][k] += value
            hits = matches(selected['current'], targets)[0]
            room = len(selected['current']) < len(record['trial']['primary']) + 5
            missing = []
            for ti in sorted(set(range(len(targets))) - hits):
                target = targets[ti]
                support = any(p['class_id'] == target['class_id'] and p['confidence'] > .25 and
                              box_iou(p['box_xyxy'], target['box']) >= .5 for p in teacher['predictions']['merged_predictions'])
                cause = ('no_teacher_support_geometry' if not support else
                         'shared_extra_budget_full' if not room else 'eligible_support_but_no_strict_complement')
                causes[cause] += 1
                missing.append(dict(target_index=ti, class_id=target['class_id'], cause=cause))
            capacity['global_ten_box_oracle_misses'] += max(0, len(targets)-10)
            capacity['fixed_primary_plus_five_extra_oracle_misses'] += max(0, len(targets)-len(record['trial']['primary'])-5)
            rows.append(dict(image=entry['image'], metrics=metrics, missing=missing, budget_room=room))
        assert totals['current'] == report['summary']['trial']
        assert totals['feature'] == report['summary']['current']
        assert sum(causes.values()) == totals['current']['fn']
        summary[stage] = dict(images=count, totals=totals, remaining_miss_causes=causes, capacity_oracles=capacity,
                              warning='Miss causes use GT post-selection, never as policy inputs; oracles are limits, not achieved accuracy')
        save(OUT / (stage + '.json'), dict(summary=summary[stage], cases=rows))
    assert {p: sha(Path(p)) for p in pins} == pins and resolution_runtime_fingerprint(REPO) == frozen
    save(OUT / 'report.json', dict(status='complete', summary=summary, pins=pins, runtime_fingerprint=frozen,
         new_model_inference=False, source_cached=True, validation_reused=True, field_accuracy=False,
         scope='Port GT class-aware IoU0.5 localization, not electrical faults or cross-cabinet/continuity'))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
