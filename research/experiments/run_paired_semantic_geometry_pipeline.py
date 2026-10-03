"""Watch only the owned finite feature job then run fixed existing gates."""
import ctypes
import subprocess
import sys
import time
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_semantic_geometry import ROOT,NEW,load,save,sha


def process_is_running(pid):
    from ctypes import wintypes
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=kernel.OpenProcess(0x1000,False,pid)
    if not handle:return False
    try:
        code=wintypes.DWORD()
        if not kernel.GetExitCodeProcess(handle,ctypes.byref(code)):raise OSError(ctypes.get_last_error())
        return code.value==259
    finally:kernel.CloseHandle(handle)


def main():
    progresspath=NEW/'pipeline_progress.json'
    if progresspath.exists():raise FileExistsError('No duplicate enriched pipeline')
    script=ROOT/'experiments/run_paired_semantic_geometry_stage.py'
    paths=[Path(__file__),script,ROOT/'artifacts/paired_semantic_localization_preregistration_20261003/PLAN.md']
    paths += [ROOT/'experiments'/name for name in ('prepare_paired_semantic_geometry.py','paired_semantic_geometry_examples.py',
        'train_paired_port_semantic_heads.py','evaluate_paired_port_semantic_train.py','train_paired_port_semantic_full.py',
        'evaluate_paired_port_semantic_holdouts.py','paired_port_semantic_selection.py','paired_port_semantics.py','port_semantic_verifier.py')]
    pins={str(p):sha(p) for p in paths};initial=load(NEW/'features_train/progress.json');owner_pid=initial.get('pid')
    save(NEW/'pipeline_protocol.json',dict(pins=pins,owned_feature_pid=owner_pid,classifier_OOF_only=True,
        exact_existing_gates=True,metadata_fold_derived_from_fixed_source_order=True,no_score_or_GT_edits=True,
        no_automatic_deployment=True,field_accuracy=False))
    started=time.monotonic()
    while not (NEW/'features_train/report.json').exists():
        state=load(NEW/'features_train/progress.json')
        if state['status']=='failed' or (owner_pid and not process_is_running(owner_pid)):
            save(progresspath,dict(status='failed_feature_preparation',feature_progress=state));return
        if time.monotonic()-started>7200:
            save(progresspath,dict(status='feature_wait_timeout',feature_process_not_stopped=True));return
        save(progresspath,dict(status='running',phase='owned_train_geometry_feature_preparation',feature_pid=owner_pid,
                               completed=state.get('completed'),total=state.get('total')))
        time.sleep(2)
    for phase in ('classifier','source','full','holdouts'):
        assert {p:sha(Path(p)) for p in pins}==pins
        with (NEW/(phase+'.log')).open('wb') as log:
            child=subprocess.Popen([sys.executable,'-B',str(script),phase],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            save(progresspath,dict(status='running',phase=phase,pid=child.pid));code=child.wait()
        if code:save(progresspath,dict(status='failed',phase=phase,return_code=code));return
        if phase=='classifier' and not load(NEW/'heads_oof/report.json')['qualifies_crop_feasibility']:
            save(progresspath,dict(status='rejected_crop',summary=load(NEW/'heads_oof/report.json')['summary'],source_not_run=True));return
        if phase=='source' and not load(NEW/'source_train_oof/report.json')['qualifies']:
            save(progresspath,dict(status='rejected_OOF_source',summary=load(NEW/'source_train_oof/report.json')['summary'],holdouts_not_run=True));return
    save(progresspath,dict(status='complete',acceptance=load(NEW/'holdouts/progress.json'),no_automatic_deployment=True))


if __name__=='__main__':main()
