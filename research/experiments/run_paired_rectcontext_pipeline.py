"""Finite rectangle feature/OOF/source job with one predeclared control."""
import os
import sys
from pathlib import Path
sys.dont_write_bytecode = True
from prepare_paired_rectcontext import OUT, main as prepare
from prepare_paired_port_semantics import ROOT, REPO, load, save, sha
import run_paired_rectcontext_stage as gates
import evaluate_paired_rectcontext_full_train as full_gate
import prepare_paired_rectcontext_control as control
from inspection_agent.paired_port_geometry import paired_runtime_fingerprint


def progress(phase, **fields):
    record = dict(status='running', pid=os.getpid(), phase=phase); record.update(fields)
    save(OUT / 'pipeline_progress.json', record)


def run_gates(destination):
    old_out, old_full = gates.OUT, full_gate.OUT
    try:
        gates.OUT = full_gate.OUT = destination
        progress('fixed400_classifier_OOF', experiment=str(destination)); gates.execute('classifier')
        if not load(destination / 'heads_oof/report.json')['qualifies_crop_feasibility']:
            return 'crop_rejected'
        progress('ALL192_source_OOF_against_stronger289', experiment=str(destination)); gates.execute('source')
        if not load(destination / 'source_train_oof/report.json')['qualifies']:
            return 'source_rejected'
        progress('fixed_full_head', experiment=str(destination)); gates.execute('full')
        progress('exact_full_head_TRAIN192_against289', experiment=str(destination)); full_gate.main()
        if not load(destination / 'full_train/report.json')['qualifies']:
            return 'full_train_rejected'
        progress('fresh_inner48_then_outer30_against65_33', experiment=str(destination)); gates.execute('holdouts')
        result = load(destination / 'holdouts/progress.json')
        return 'holdouts_rejected' if result['status'] == 'rejected' else 'source_pass_requires_reference_ROI_SAM_Qt'
    finally:
        gates.OUT, full_gate.OUT = old_out, old_full


def main():
    if (OUT / 'pipeline_progress.json').exists(): raise FileExistsError('Preserve finite rectangle run')
    assert load(OUT / 'smoke/report.json')['real_rectcontext_inference_passed']
    files = [Path(__file__)] + [Path(__file__).with_name(name) for name in (
        'prepare_paired_rectcontext.py', 'paired_rectcontext_features.py', 'run_paired_rectcontext_stage.py',
        'evaluate_paired_rectcontext_full_train.py', 'prepare_paired_rectcontext_control.py',
        'train_paired_port_semantic_heads.py', 'evaluate_paired_port_semantic_train.py',
        'train_paired_port_semantic_full.py', 'evaluate_paired_port_semantic_holdouts.py',
        'port_semantic_verifier.py', 'paired_port_semantic_selection.py')]
    files += [ROOT / 'artifacts/paired_rectcontext_preregistration_20261003/PLAN.md',
              ROOT / 'artifacts/paired_rectcontext_stronger_baseline_preregistration_20261003/PLAN.md']
    pins = {str(p): sha(p) for p in files}; frozen = paired_runtime_fingerprint(REPO)
    gates.prepare_baseline()
    save(OUT / 'pipeline_protocol.json', dict(pins=pins, accepted_paired_runtime=frozen,
        stronger_baseline_tp=[289, 65, 33], baseline_unmatched=[4, 0, 1],
        aspect4166_then_one3134_crop_only_control=True, no_automatic_deployment=True, field_accuracy=False))
    try:
        progress('fresh_TRAIN192_rectangular_pixels'); prepare('full')
        result = run_gates(OUT); outcomes = dict(aspect4166=result)
        if result == 'crop_rejected':
            progress('predeclared_no_aspect3134_control'); control.main()
            outcomes['original3134'] = run_gates(control.OUT)
        assert {p: sha(Path(p)) for p in pins} == pins and paired_runtime_fingerprint(REPO) == frozen
        status = 'source_pass_requires_reference_ROI_SAM_Qt' if any(v.startswith('source_pass') for v in outcomes.values()) else 'rejected'
        progress('finished', status=status, outcomes=outcomes, no_automatic_deployment=True)
    except BaseException as error:
        progress('failed', status='failed', error=type(error).__name__ + ': ' + str(error)); raise


if __name__ == '__main__': main()
