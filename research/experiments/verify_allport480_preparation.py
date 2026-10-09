"""Offline software preparation checks, not detector/field accuracy acceptance."""
import sys
import unittest
import subprocess
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,REPO,load,save,sha
from inspection_agent.paired_native_pose import native_pose_runtime_fingerprint

OUT=ROOT/'artifacts/allport480_continuation_software_checks_20261005'


def main(output=OUT):
    global OUT
    OUT=Path(output).resolve()
    if OUT.exists():raise FileExistsError('preserve software verification')
    protocol=load(ROOT/'artifacts/allport480_source_20261005/protocol.json');frozen=native_pose_runtime_fingerprint(REPO)
    dirty=lambda:subprocess.check_output(['E:/Git/cmd/git.exe','-C',str(REPO),'status','--porcelain'],text=True)
    before=dirty()
    if frozen!=protocol['runtime'] or before!=protocol['mainline_dirty_before']:raise ValueError('protect live mainline/dirty state')
    names=['test_replacement_voters','test_allport480_source_audit','test_exact_paired_score_cache','test_allport480_misses','test_allport480_teacher_holdouts','test_protected_allport_baseline']
    suite=unittest.defaultTestLoader.loadTestsFromNames(names)
    OUT.mkdir()
    with (OUT/'tests.log').open('w',encoding='utf-8') as stream:
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    if not result.wasSuccessful():raise RuntimeError('prepared software checks failed')
    if native_pose_runtime_fingerprint(REPO)!=frozen or dirty()!=before:raise ValueError('software checks changed live mainline')
    pins={str(p):sha(p) for p in [Path(__file__),*[Path(__file__).with_name(n+'.py') for n in names],
        *[Path(__file__).with_name(n+'.py') for n in ['exact_paired_score_cache','audit_allport480_source','diagnose_allport480_misses',
            'run_allport480_teacher_source','run_allport480_teacher_holdouts','audit_allport480_teacher_holdouts','protected_allport_baseline','recover_allport480_summary']]]}
    save(OUT/'report.json',dict(status='pass',tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),pins=pins,
        runtime=frozen,mainline_dirty_state_unchanged=True,no_detector_inference=True,no_training=True,
        no_development_inputs_read=True,no_deployment=True,field_accuracy=None,
        warning='Software/cache/gate tests only; does not complete running source192 or demonstrate improved cabinet recognition.'))
    print(dict(status='pass',tests=result.testsRun,mainline_unchanged=True,field_accuracy=None))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=OUT)
    main(parser.parse_args().output)
