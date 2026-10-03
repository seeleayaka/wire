"""Scoped fixed-head gate: only swap GT-free median proposal inputs."""
import copy
import sys
from pathlib import Path
sys.dont_write_bytecode = True
import evaluate_paired_multimodel_head as gate
from paired_median_proposals import proposals
from prepare_paired_port_semantics import ROOT, REPO, load, save, sha
from prepare_paired_semantic_geometry import NEW as GEOMETRY
from current_port_baseline_audit import BASE, read_current_case
from inspection_agent.port_tiling import box_iou
OUT = ROOT / 'artifacts/paired_median_current_head_20261003'
AUDIT = ROOT / 'artifacts/paired_median_train_ceiling_20261003'
PREP = ROOT / 'artifacts/paired_support_graph_20261003/source_selections'
PROXY = ROOT / 'artifacts/paired_median_scoped_input_20261003'


def main():
    if OUT.exists(): raise FileExistsError('Preserve frozen-head median trial')
    old_out, old_proposals, old_load, old_save = gate.OUT, gate.PROPOSALS, gate.load, gate.save
    extra_pins = {str(p): sha(p) for p in (Path(__file__), Path(__file__).with_name('paired_median_proposals.py'),
        ROOT / 'artifacts/paired_median_head_preregistration_20261003/PLAN.md', AUDIT / 'report.json')}
    def adapted_load(path):
        path = Path(path)
        if path == PROXY / 'report.json':
            report = copy.deepcopy(old_load(AUDIT / 'report.json'))
            report['pins'].update(extra_pins)
            return report
        if path.parent == PROXY and path.name.endswith('_proposals.json'):
            stage, stem = path.name[:-len('_proposals.json')].split('_', 1)
            if stage == 'train':
                cached = AUDIT / (stem + '_proposals.json')
                extra_pins[str(cached)] = sha(cached)
                result = old_load(cached)
                return dict(image=result['image'], current=result['current'], extended_candidates=result['candidates'])
            name = stem + '.JPG'; index_path = PREP / stage / 'index.json'
            extra_pins[str(index_path)] = sha(index_path)
            indexed = next(r for r in old_load(index_path)['records'] if r['image'] == name)
            cache_path = Path(indexed['path']); assert sha(cache_path) == indexed['sha256']; extra_pins[str(cache_path)] = sha(cache_path)
            case = old_load(cache_path); reportpath = BASE / stage / 'report.json'; extra_pins[str(reportpath)] = sha(reportpath)
            entry = next(r for r in old_load(reportpath)['cases'] if r['image'] == name)
            teacher, baseline = read_current_case(stage, entry, extra_pins); assert teacher == case['teacher']
            accepted = GEOMETRY / 'holdouts' / stage / (stem + '_predictions.json')
            extra_pins[str(accepted)] = sha(accepted); current = old_load(accepted)['trial']
            remaining = 5 - (len(current['all_predictions']) - len(current['primary'])); assert remaining >= 0
            rows = proposals(teacher, [teacher, case['student'], case['feature'], baseline['alternative']]) if remaining else []
            rows = [r for r in rows if not any(box_iou(r['box_xyxy'], q['box_xyxy']) >= .5 for q in current['all_predictions'])]
            result = dict(image=name, current=current, extended_candidates=rows, GT_not_used=True)
            directory = OUT / 'proposal_cache'; directory.mkdir(exist_ok=True)
            old_save(directory / path.name, result)
            return result
        return old_load(path)
    def adapted_save(path, value):
        if Path(path).name in ('protocol.json', 'report.json') and 'pins' in value:
            value = copy.deepcopy(value); value['pins'].update(extra_pins)
            value.update(median_geometry=True, fixed_accepted_head=True, scoped_adapter=True)
        old_save(path, value)
    # Virtual proposal input hashes map to actual train files or deterministic
    # cache files; gate.sha must never hash absent proxy paths.
    old_sha = gate.sha
    def adapted_sha(path):
        path = Path(path)
        if path == PROXY / 'report.json': return old_sha(AUDIT / 'report.json')
        if path.parent == PROXY and path.name.endswith('_proposals.json'):
            stage, stem = path.name[:-len('_proposals.json')].split('_', 1)
            actual = AUDIT / (stem + '_proposals.json') if stage == 'train' else OUT / 'proposal_cache' / path.name
            if stage != 'train' and not actual.exists(): adapted_load(path)
            return old_sha(actual)
        return old_sha(path)
    try:
        gate.OUT, gate.PROPOSALS, gate.load, gate.save, gate.sha = OUT, PROXY, adapted_load, adapted_save, adapted_sha
        gate.main()
        assert {p: old_sha(Path(p)) for p in extra_pins} == extra_pins
    finally:
        gate.OUT, gate.PROPOSALS, gate.load, gate.save, gate.sha = old_out, old_proposals, old_load, old_save, old_sha


if __name__ == '__main__': main()
