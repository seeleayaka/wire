"""Finite GT-free pose proposals, unchanged fixed head, stronger median prefix."""
import copy
import sys
from pathlib import Path
sys.dont_write_bytecode = True
from prepare_paired_port_semantics import ROOT, REPO, load, save, sha
import evaluate_paired_multimodel_head as gate
from current_port_baseline_audit import BASE, read_current_case
from paired_pose_search import proposals, select
from inspection_agent.paired_median_geometry import median_runtime_fingerprint

OUT = ROOT / 'artifacts/paired_pose_search_20261003'
PREP = ROOT / 'artifacts/paired_support_graph_20261003/source_selections'
MEDIAN = ROOT / 'artifacts/paired_median_current_head_20261003'
PLAN = ROOT / 'artifacts/paired_pose_search_preregistration_20261003/PLAN.md'


def prepare_inputs(folder):
    folder.mkdir(parents=True)
    pins = {}
    for path in (Path(__file__), Path(__file__).with_name('paired_pose_search.py'),
                 Path(__file__).with_name('paired_median_proposals.py'),
                 Path(__file__).with_name('paired_positive_jitter_examples.py'), PLAN,
                 ROOT / 'artifacts/paired_median_complete_sam_20261003/acceptance.json',
                 ROOT / 'artifacts/paired_median_source_audit_20261003/report.json',
                 REPO / 'inspection_agent/paired_median_geometry.py',
                 REPO / 'inspection_agent/paired_port_median_features.py',
                 REPO / 'config/paired_median_geometry_20261003.json'):
        pins[str(path)] = sha(path)
    assert load(ROOT / 'artifacts/paired_median_complete_sam_20261003/acceptance.json')['status'] == 'complete'
    assert load(ROOT / 'artifacts/paired_median_source_audit_20261003/report.json')['source_gates_passed']
    counts = {}
    for stage, expected in (('train', 192), ('inner', 48), ('outer', 30)):
        index_path = PREP / stage / 'index.json'
        pins[str(index_path)] = sha(index_path)
        indexed = {r['image']: r for r in load(index_path)['records']}
        report_path = BASE / stage / 'report.json'
        pins[str(report_path)] = sha(report_path)
        entries = load(report_path)['cases']; assert len(entries) == expected
        strong_report = MEDIAN / stage / 'report.json'; pins[str(strong_report)] = sha(strong_report)
        counts[stage] = 0
        for entry in entries:
            name = entry['image']; path = Path(indexed[name]['path'])
            assert sha(path) == indexed[name]['sha256']; pins[str(path)] = sha(path)
            case = load(path); teacher, old = read_current_case(stage, entry, pins)
            assert teacher == case['teacher']
            current_path = MEDIAN / stage / (Path(name).stem + '_predictions.json')
            pins[str(current_path)] = sha(current_path); current = load(current_path)['trial']
            remaining = 5 - (len(current['all_predictions']) - len(current['primary']))
            assert remaining >= 0
            rows = proposals(teacher, [teacher, case['student'], case['feature'], old['alternative']], current) if remaining else []
            result = dict(image=name, current=current, extended_candidates=rows,
                GT_not_used=True, one_alert_per_generating_parent=True, remaining_slots=remaining)
            save(folder / (stage + '_' + Path(name).stem + '_proposals.json'), result)
            counts[stage] += len(rows)
    assert {p: sha(Path(p)) for p in pins} == pins
    save(folder / 'report.json', dict(status='complete', pins=pins, counts=counts,
        GT_not_used=True, stronger_baseline=[289, 65, 33], fixed_accepted_head=True))
    return pins


def execute():
    if OUT.exists(): raise FileExistsError('Preserve existing finite pose evaluation')
    inputs = ROOT / 'artifacts/paired_pose_search_inputs_20261003'
    if inputs.exists(): raise FileExistsError('Preserve existing native pose proposals')
    frozen = median_runtime_fingerprint(REPO)
    pins = prepare_inputs(inputs)
    prior = gate.OUT, gate.PROPOSALS, gate.load, gate.save, gate.select
    old_out, old_proposals, old_load, old_save, old_select = prior
    def adapted_load(path):
        path = Path(path)
        if path == gate.GEOMETRY / 'full_train/report.json':
            return old_load(MEDIAN / 'train/report.json')
        for stage in ('inner', 'outer'):
            if path == gate.GEOMETRY / 'holdouts' / stage / 'report.json':
                return old_load(MEDIAN / stage / 'report.json')
        return old_load(path)
    def adapted_save(path, value):
        if Path(path).name in ('protocol.json', 'report.json') and 'pins' in value:
            value = copy.deepcopy(value); value['pins'].update(pins)
            value.update(uniform_pose_search=True, actual_input_paths=True,
                median_runtime_fingerprint=frozen, same_head_no_training=True)
        old_save(path, value)
    try:
        gate.OUT, gate.PROPOSALS, gate.load, gate.save, gate.select = OUT, inputs, adapted_load, adapted_save, select
        gate.main()
        assert median_runtime_fingerprint(REPO) == frozen
    finally:
        gate.OUT, gate.PROPOSALS, gate.load, gate.save, gate.select = prior


if __name__ == '__main__': execute()
