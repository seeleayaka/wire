"""Software-only checks and E mainline preservation. Never launches inference."""
import sys
import subprocess
import unittest
from pathlib import Path
sys.dont_write_bytecode = True
from prepare_paired_port_semantics import ROOT, REPO, load, save, sha
from inspection_agent.paired_native_pose import native_pose_runtime_fingerprint

OUT = ROOT/'artifacts/allport_common640_software_preparation_20261005'


def main():
    if OUT.exists(): raise FileExistsError('preserve preparation verification')
    source = load(ROOT/'artifacts/allport480_teacher_source_20261005/report.json')
    source_protocol = load(ROOT/'artifacts/allport480_teacher_source_20261005/protocol.json')
    frozen = native_pose_runtime_fingerprint(REPO)
    dirty = lambda: subprocess.check_output(['E:/Git/cmd/git.exe', '-C', str(REPO), 'status', '--porcelain'], text=True)
    before = dirty()
    if frozen != source['runtime'] or before != source_protocol['mainline_dirty_before']:
        raise ValueError('preserve live mainline/runtime state')
    names = ['test_common_scale640', 'test_common_scale_evidence', 'test_common_scale_source_gate']
    suite = unittest.defaultTestLoader.loadTestsFromNames(names)
    OUT.mkdir()
    with (OUT/'tests.log').open('w', encoding='utf-8') as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    if not result.wasSuccessful(): raise RuntimeError('unlaunched preparation tests failed')
    if native_pose_runtime_fingerprint(REPO) != frozen or dirty() != before:
        raise ValueError('software verification changed mainline')
    files = [Path(__file__), ROOT/'artifacts/allport_common640_preregistration_20261005/PLAN.md']
    files += [Path(__file__).with_name(n+'.py') for n in names+['paired_common_scale640', 'common_scale_evidence']]
    save(OUT/'report.json', dict(status='pass', tests=result.testsRun, pins={str(p): sha(p) for p in files},
         runtime=frozen, mainline_dirty_state_unchanged=True, no_model_inference=True, no_training=True,
         no_development_pixels_or_labels_read=True, no_deployment=True, field_accuracy=None,
         runner_and_full_source_auditor_preparation_complete=False,
         warning='Software tests only. Current fresh INNER worker remains active; this common640 experiment has NOT run.'))
    print(dict(status='pass', tests=result.testsRun, no_new_inference=True, no_deployment=True))


if __name__ == '__main__': main()
