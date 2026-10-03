"""Exact generic proposal parity between staged portable and tested helper."""
import importlib.util
import sys
from pathlib import Path
sys.dont_write_bytecode = True
from prepare_paired_port_semantics import ROOT, REPO, load, save, sha
from current_port_baseline_audit import BASE, read_current_case
from paired_median_proposals import proposals
from inspection_agent.paired_port_geometry import paired_runtime_fingerprint
OUT = ROOT / 'artifacts/paired_median_helper_parity_20261003'
STAGED = ROOT / 'staging/paired_median_release/paired_port_median_features.py'
PREP = ROOT / 'artifacts/paired_support_graph_20261003/source_selections'


def main():
    if OUT.exists(): raise FileExistsError('Preserve portable helper checks')
    pins = {str(p): sha(p) for p in (Path(__file__), STAGED, Path(__file__).with_name('paired_median_proposals.py'))}
    frozen = paired_runtime_fingerprint(REPO)
    spec = importlib.util.spec_from_file_location('median_portable_parity_helper', STAGED)
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    count = total = 0
    for stage, expected in (('train', 192), ('inner', 48), ('outer', 30)):
        path = PREP / stage / 'index.json'; pins[str(path)] = sha(path); rows = load(path)['records']; assert len(rows) == expected
        entries = {r['image']: r for r in load(BASE / stage / 'report.json')['cases']}
        for item in rows:
            path = Path(item['path']); assert sha(path) == item['sha256']; pins[str(path)] = sha(path); case = load(path)
            teacher, baseline = read_current_case(stage, entries[item['image']], pins); assert teacher == case['teacher']
            models = [teacher, case['student'], case['feature'], baseline['alternative']]
            current = proposals(teacher, models); actual = helper.proposals(teacher, models)
            assert current == actual; count += 1; total += len(actual)
    assert count == 270 and {p: sha(Path(p)) for p in pins} == pins and paired_runtime_fingerprint(REPO) == frozen
    OUT.mkdir(); save(OUT / 'report.json', dict(status='complete', source_count=count, raw_proposals=total,
        all_native_dictionary_parity=True, pins=pins, GT_not_read=True, field_accuracy=False, no_deployment=True))
    print(str(dict(source_count=count, raw_proposals=total, parity=True)), flush=True)


if __name__ == '__main__': main()
