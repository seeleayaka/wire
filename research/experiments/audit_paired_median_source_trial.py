"""Resolve scoped virtual inputs to pinned real files and recheck all270."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
from prepare_paired_port_semantics import ROOT, REPO, load, save, sha
from prepare_paired_semantic_geometry import NEW as GEOMETRY
from current_port_baseline_audit import BASE, read_targets
from run_paired_median_current_head import OUT as TRIAL, AUDIT, PROXY
from paired_port_semantic_selection import select
from audit_port_multiscale_acceptance import metric, matches
from inspection_agent.paired_port_geometry import paired_runtime_fingerprint, HEAD_SHA
OUT = ROOT / 'artifacts/paired_median_source_audit_20261003'


def main():
    if OUT.exists(): raise FileExistsError('Preserve source audit')
    report = load(TRIAL / 'report.json'); protocol = load(TRIAL / 'protocol.json')
    frozen = paired_runtime_fingerprint(REPO); assert frozen == protocol['runtime_fingerprint']
    assert report['status'] == 'source_pass_requires_reference_ROI_SAM_Qt'
    pins = {}; aliases = {}
    for virtual, digest in report['pins'].items():
        path = Path(virtual)
        if path.parent == PROXY:
            if path.name == 'report.json': actual = AUDIT / 'report.json'
            else:
                stage, stem = path.name[:-len('_proposals.json')].split('_', 1)
                actual = AUDIT / (stem + '_proposals.json') if stage == 'train' else TRIAL / 'proposal_cache' / path.name
            aliases[virtual] = str(actual); path = actual
        assert sha(path) == digest, str(path)
        if str(path) in pins: assert pins[str(path)] == digest
        pins[str(path)] = digest
    for path in (Path(__file__), TRIAL / 'protocol.json', TRIAL / 'report.json'):
        pins[str(path)] = sha(path)
    records = []; summaries = {}
    for stage, count in (('train', 192), ('inner', 48), ('outer', 30)):
        entries = load(BASE / stage / 'report.json')['cases']; assert len(entries) == count
        for entry in entries:
            name = entry['image']; path = TRIAL / stage / (Path(name).stem + '_predictions.json')
            pins[str(path)] = sha(path); saved = load(path)
            accepted_path = (GEOMETRY / 'full_train' if stage == 'train' else GEOMETRY / 'holdouts' / stage) / path.name
            pins[str(accepted_path)] = sha(accepted_path); assert saved['current'] == load(accepted_path)['trial']
            predicted = select(saved['current'], saved['proposals'], saved['probabilities'], HEAD_SHA)
            assert predicted == saved['trial']
            assert all(p.get('median_geometry') and len(p.get('median_support_boxes', {})) >= 2 for p in saved['proposals'])
            targets = read_targets(stage, name, [2736, 3648], entry['label_sha256'], pins)
            old, new = saved['current']['all_predictions'], saved['trial']['all_predictions']
            oh, nh = matches(old, targets)[0], matches(new, targets)[0]
            row = dict(stage=stage, image=name, current=metric(old, targets), trial=metric(new, targets),
                lost=sorted(oh - nh), gained=sorted(nh - oh), additions=len(saved['trial']['paired_semantic_additions']))
            assert not row['lost'] and row['trial']['unmatched'] <= row['current']['unmatched']
            if name.startswith('normal_'): assert not new
            records.append(row)
        local = [r for r in records if r['stage'] == stage]
        totals = {kind: {key: sum(r[kind][key] for r in local) for key in ('tp', 'unmatched', 'fn', 'predictions', 'targets')} for kind in ('current', 'trial')}
        assert totals == report['stages'][stage]['summary']
        assert report['stages'][stage]['qualifies']
        summaries[stage] = totals
    assert {p: sha(Path(p)) for p in pins} == pins and paired_runtime_fingerprint(REPO) == frozen
    OUT.mkdir(); result = dict(status='complete', source_gates_passed=True, stages=summaries, cases=records, pins=pins,
        explicit_virtual_input_aliases=aliases, all270_selection_and_scoring_rechecked=True,
        same_fixed_head=True, real_workflow_pending=True, validation_reused=True, field_accuracy=False)
    save(OUT / 'report.json', result); print(str(dict(stages=summaries, aliases=len(aliases))), flush=True)


if __name__ == '__main__': main()
