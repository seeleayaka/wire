"""Finite preregistered positive-jitter run, never automatically deploy."""
import os
import sys
from pathlib import Path
sys.dont_write_bytecode = True
from prepare_paired_positive_jitter import OUT, main as prepare
from prepare_paired_port_semantics import ROOT, REPO, load, save, sha
from run_paired_positive_jitter_stage import execute, exact_full_train
from inspection_agent.paired_port_geometry import paired_runtime_fingerprint


def progress(phase, **fields):
    record = dict(status='running', pid=os.getpid(), phase=phase)
    record.update(fields)
    save(OUT / 'pipeline_progress.json', record)


def main():
    if (OUT / 'pipeline_progress.json').exists():
        raise FileExistsError('Preserve finite positive-jitter run')
    assert load(OUT / 'smoke/report.json')['fresh_jitter_inference_passed']
    assert load(ROOT / 'artifacts/paired_geometry_complete_sam_20261003/acceptance.json')['sam_complete']
    files = [Path(__file__)] + [Path(__file__).with_name(name) for name in (
        'prepare_paired_positive_jitter.py', 'paired_positive_jitter_examples.py',
        'run_paired_positive_jitter_stage.py', 'run_paired_boxpool_stage.py',
        'evaluate_paired_boxpool_full_train.py', 'train_paired_port_semantic_heads.py',
        'evaluate_paired_port_semantic_train.py', 'train_paired_port_semantic_full.py',
        'evaluate_paired_port_semantic_holdouts.py', 'port_semantic_verifier.py',
        'paired_port_semantic_selection.py')]
    files += [ROOT / 'artifacts/paired_positive_jitter_preregistration_20261003/PLAN.md']
    pins = {str(path): sha(path) for path in files}
    frozen = paired_runtime_fingerprint(REPO)
    save(OUT / 'pipeline_protocol.json', dict(pins=pins, accepted_paired_runtime=frozen,
        baseline_tp=[286, 64, 33], baseline_unmatched=[4, 0, 1],
        original_central_context_embeddings=True, no_automatic_deployment=True,
        no_validation_training=True, field_accuracy=False))
    try:
        progress('fresh_TRAIN192_positive_jitter_and_normal_reference'); prepare('full')
        progress('fixed400_classifier_OOF'); execute('classifier')
        if not load(OUT / 'heads_oof/report.json')['qualifies_crop_feasibility']:
            progress('fixed400_classifier_OOF', status='rejected'); return
        progress('ALL192_source_OOF_against_accepted286'); execute('source')
        if not load(OUT / 'source_train_oof/report.json')['qualifies']:
            progress('ALL192_source_OOF_against_accepted286', status='rejected'); return
        progress('fixed_full_head'); execute('full')
        progress('exact_full_head_TRAIN192_against_accepted286'); exact_full_train()
        if not load(OUT / 'full_train/report.json')['qualifies']:
            progress('exact_full_head_TRAIN192_against_accepted286', status='rejected'); return
        progress('fresh_inner48_then_outer30'); execute('holdouts')
        acceptance = load(OUT / 'holdouts/progress.json')
        assert {p: sha(Path(p)) for p in pins} == pins
        assert paired_runtime_fingerprint(REPO) == frozen
        progress('finished', status='rejected' if acceptance['status'] == 'rejected' else 'complete',
                 acceptance=acceptance, no_automatic_deployment=True)
    except BaseException as error:
        progress('failed', status='failed', error=type(error).__name__ + ': ' + str(error))
        raise


if __name__ == '__main__':
    main()
