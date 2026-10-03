"""Create immutable release provenance only AFTER all real live checks pass."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
from prepare_paired_port_semantics import ROOT, REPO, load, save, sha
from inspection_agent.paired_port_geometry import paired_runtime_fingerprint, HEAD_SHA, ENCODER_SHA
OUT = ROOT / 'staging/paired_median_release'


def main():
    destination = OUT / 'paired_median_geometry_20261003.json'
    if destination.exists(): raise FileExistsError('Preserve release manifest')
    live_folder = ROOT / 'artifacts/paired_median_live_20261003'
    live = load(live_folder / 'report.json'); protocol = load(live_folder / 'protocol.json')
    assert live['qualifies_live_diagnostic'] and live['gains'] == dict(train=3, inner=1, outer=0)
    assert len(live['cases']) == len(protocol['categories']) == 22
    assert all(not r['lost'] and r['accepted'] == r['expected_candidates'] and r['trial']['unmatched'] <= r['current']['unmatched'] for r in live['cases'])
    frozen = paired_runtime_fingerprint(REPO); assert frozen == protocol['accepted_paired_runtime']
    assert {p: sha(Path(p)) for p in protocol['pins']} == protocol['pins']
    source_path = ROOT / 'artifacts/paired_median_source_audit_20261003/report.json'; source = load(source_path)
    assert source['source_gates_passed'] and {p: sha(Path(p)) for p in source['pins']} == source['pins']
    helper_path = OUT / 'paired_port_median_features.py'; parity_path = ROOT / 'artifacts/paired_median_helper_parity_20261003/report.json'
    parity = load(parity_path); assert parity['all_native_dictionary_parity'] and parity['source_count'] == 270
    assert parity['pins'][str(helper_path)] == sha(helper_path)
    manifest = dict(policy_id='accepted_paired_preserved_median_geometry_v1_20261003',
        head_sha256=HEAD_SHA, encoder_sha256=ENCODER_SHA, geometry_sha256=sha(helper_path), accepted_runtime=frozen,
        probability_gate=.98, feature_dimensions=6144, classes=['other', 'unplugged_plug', 'unplugged_jack'], contexts=[1.5, 3.],
        minimum_unique_checkpoint_votes=2, support_iou=.5, proposal_floor=.05, complete_margin=16,
        maximum_primary=5, maximum_extra=5, source_gates_passed=True, live_diagnostic_passed=True,
        source_evaluation_sha256=sha(source_path), live_evaluation_sha256=sha(live_folder / 'report.json'),
        source_tp=[289, 65, 33], source_targets=[344, 80, 56], source_unmatched=[4, 0, 1],
        same_fixed_head_no_retraining=True, default_off=True, manual_review_only=True,
        complete_SAM_Qt_acceptance_required=True, validation_reused=True, field_accuracy=False,
        physical_fault_confirmation=False, automatic_fault_verdict=False)
    save(destination, manifest); print(str(dict(path=str(destination), sha256=sha(destination))), flush=True)


if __name__ == '__main__': main()
