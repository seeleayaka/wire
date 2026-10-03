"""Original fixed gates, rectangular embeddings, stronger median prefix."""
import argparse
import copy
import importlib
import sys
from pathlib import Path
sys.dont_write_bytecode = True
from prepare_paired_rectcontext import OUT
from prepare_paired_port_semantics import ROOT, REPO, load, save, sha
from paired_rectcontext_features import embeddings
from run_paired_boxpool_stage import MODULES
from inspection_agent.paired_port_geometry import paired_runtime_fingerprint
TRIAL = ROOT / 'artifacts/paired_median_current_head_20261003'
BASELINE = ROOT / 'artifacts/paired_rectcontext_stronger_baseline_20261003'


def prepare_baseline():
    if BASELINE.exists():
        protocol = load(BASELINE / 'protocol.json'); assert {p: sha(Path(p)) for p in protocol['pins']} == protocol['pins']; return
    from current_port_baseline_audit import BASE
    audit_path = ROOT / 'artifacts/paired_median_source_audit_20261003/report.json'
    audit = load(audit_path); assert audit['source_gates_passed']
    pins = dict(audit['pins']); pins[str(audit_path)] = sha(audit_path)
    BASELINE.mkdir()
    for stage, expected in (('train', 289), ('inner', 65), ('outer', 33)):
        path = BASE / stage / 'report.json'; pins[str(path)] = sha(path); report = copy.deepcopy(load(path))
        fixed = TRIAL / stage / 'report.json'; pins[str(fixed)] = sha(fixed); measured = load(fixed)
        assert measured['qualifies'] and measured['summary']['trial']['tp'] == expected
        by_name = {r['image']: r for r in measured['cases']}
        for row in report['cases']: row['trial'] = copy.deepcopy(by_name[row['image']]['trial'])
        report['summary']['trial'] = copy.deepcopy(measured['summary']['trial'])
        folder = BASELINE / stage; folder.mkdir(); save(folder / 'report.json', report)
    assert {p: sha(Path(p)) for p in pins} == pins
    save(BASELINE / 'protocol.json', dict(pins=pins, stronger_median_source_baseline=True,
        counts=[289, 65, 33], unmatched=[4, 0, 1], prospective_median_SAM_pending=True,
        no_GT_selected_runtime_proposals=True, no_deployment=True))


def execute(stage):
    prepare_baseline(); module = importlib.import_module(MODULES[stage])
    previous = module.OUT, module.load, getattr(module, 'BASE', None), getattr(module, 'embeddings', None)
    old_out, old_load, old_base, old_embeddings = previous
    frozen = paired_runtime_fingerprint(REPO)
    def adapted_load(path):
        payload = old_load(path)
        if stage in ('source', 'holdouts') and str(path).startswith(str(module.PROPOSALS)) and Path(path).name.endswith('_proposals.json'):
            split = Path(path).name.split('_', 1)[0]; name = payload['image']
            fixed = TRIAL / split / (Path(name).stem + '_predictions.json')
            payload = copy.deepcopy(payload); payload['current'] = copy.deepcopy(old_load(fixed)['trial'])
        return payload
    try:
        module.OUT, module.load = OUT, adapted_load
        if old_base is not None: module.BASE = BASELINE
        if old_embeddings is not None: module.embeddings = embeddings
        module.main(); assert paired_runtime_fingerprint(REPO) == frozen
    finally:
        module.OUT, module.load = old_out, old_load
        if old_base is not None: module.BASE = old_base
        if old_embeddings is not None: module.embeddings = old_embeddings


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=tuple(MODULES)); execute(parser.parse_args().stage)
