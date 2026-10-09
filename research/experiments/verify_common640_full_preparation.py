"""Frozen preparation regression. Does NOT launch the future source worker."""
import ast
import argparse
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
sys.dont_write_bytecode = True
from prepare_paired_port_semantics import ROOT, REPO, load, save, sha
from inspection_agent.paired_native_pose import native_pose_runtime_fingerprint

OUT = ROOT/'artifacts/allport_common640_full_software_preparation_20261005'


def main(output=OUT):
    global OUT
    OUT = Path(output).resolve()
    if OUT.exists(): raise FileExistsError('preserve preparation test evidence')
    source = load(ROOT/'artifacts/allport480_teacher_source_20261005/report.json')
    source_protocol = load(ROOT/'artifacts/allport480_teacher_source_20261005/protocol.json')
    frozen = native_pose_runtime_fingerprint(REPO)
    dirty = lambda: subprocess.check_output(['E:/Git/cmd/git.exe', '-C', str(REPO), 'status', '--porcelain'], text=True)
    before = dirty()
    if source['runtime'] != frozen or source_protocol['mainline_dirty_before'] != before:
        raise ValueError('preserve live E runtime and all dirty files')
    tests = ['test_common_scale640', 'test_common_scale_evidence', 'test_common_scale_source_gate',
             'test_common640_runner_boundary', 'test_protected_allport_baseline', 'test_exact_paired_score_cache',
             'test_replacement_voters', 'test_allport480_teacher_holdouts']
    code = ['run_allport_common640_source', 'audit_allport_common640_source', 'common_scale_evidence',
            'paired_common_scale640', 'audit_allport480_source', 'run_allport480_teacher_holdouts',
            'audit_allport480_teacher_holdouts', 'exact_paired_score_cache', 'raw_pose_consensus',
            'fine_voter_quality', 'consensus_rank', 'replacement_voters', 'protected_allport_baseline']
    files = [Path(__file__), ROOT/'artifacts/allport_common640_preregistration_20261005/PLAN.md']
    files += [Path(__file__).with_name(n+'.py') for n in code+tests]
    pins = {str(p): sha(p) for p in files}
    for p in files:
        if p.suffix == '.py': ast.parse(p.read_text(encoding='utf-8'))
    OUT.mkdir()
    # The pre-launch legacy entry tests assume no real output yet. Isolate ONLY
    # the imported runner module's OUT in this software-test process. The active
    # inference process and original filesystem directory remain untouched.
    import run_allport480_teacher_holdouts as holdouts_runner
    with tempfile.TemporaryDirectory(prefix='wire_common_qa_') as fixture:
        with patch.object(holdouts_runner, 'OUT', Path(fixture)/'never_created_holdout_fixture'):
            with (OUT/'tests.log').open('w', encoding='utf-8') as stream:
                result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(tests))
    if not result.wasSuccessful(): raise RuntimeError('prepared source/auditor software regression failed')
    if (any(sha(p) != d for p, d in pins.items()) or native_pose_runtime_fingerprint(REPO) != frozen or dirty() != before):
        raise ValueError('software verification mutated frozen preparation/mainline')
    save(OUT/'report.json', dict(status='pass', tests=result.testsRun, pins=pins, runtime=frozen,
         mainline_dirty_state_unchanged=True, no_model_inference=True, no_training=True, no_development_pixels_or_labels_read=True,
         no_deployment=True, field_accuracy=None, whole_source_runner_and_auditor_exist=True,
         source_experiment_launched=False, full_source_acceptance=False, current_validation_completed=False,
         legacy_entry_test_output_path_isolated_in_test_process=True,
         warning='Prepared software only; complete and independently audit active validation before choosing the next experiment.'))
    print(dict(status='pass', tests=result.testsRun, source_not_launched=True, E_unchanged=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, default=OUT)
    main(parser.parse_args().output)
