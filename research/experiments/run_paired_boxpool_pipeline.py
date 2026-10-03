"""One finite sequential job; stop at the first failed preregistered gate."""
import os,sys
sys.dont_write_bytecode=True
from prepare_paired_boxpool_features import OUT,main as prepare
from prepare_paired_port_semantics import ROOT,REPO,load,save,sha
from run_paired_boxpool_stage import execute
from evaluate_paired_boxpool_full_train import main as final_train


def progress(phase,**fields):
    record=dict(status='running',pid=os.getpid(),phase=phase);record.update(fields);save(OUT/'pipeline_progress.json',record)


def main():
    if (OUT/'pipeline_progress.json').exists():raise FileExistsError('Do not duplicate finite pipeline')
    if load(OUT/'smoke/report.json')['real_footprint_inference_passed'] is not True:raise ValueError('Smoke required')
    assert load(ROOT/'artifacts/paired_geometry_complete_sam_20261003/acceptance.json')['sam_complete']
    from pathlib import Path
    files=[Path(__file__),Path(__file__).with_name('prepare_paired_boxpool_features.py'),
        Path(__file__).with_name('paired_boxpool_features.py'),Path(__file__).with_name('run_paired_boxpool_stage.py'),
        Path(__file__).with_name('evaluate_paired_boxpool_full_train.py')]
    pins={str(p):sha(p) for p in files};save(OUT/'pipeline_protocol.json',dict(pins=pins,no_automatic_deployment=True))
    try:
        progress('fresh_train192_footprint_features');prepare('full')
        progress('OOF_classifier');execute('classifier')
        if not load(OUT/'heads_oof/report.json')['qualifies_crop_feasibility']:
            progress('OOF_classifier',status='rejected');return
        progress('ALL192_source_OOF');execute('source')
        if not load(OUT/'source_train_oof/report.json')['qualifies']:
            progress('ALL192_source_OOF',status='rejected');return
        progress('fixed_full_head');execute('full')
        progress('exact_full_head_TRAIN192');final_train()
        if not load(OUT/'full_train/report.json')['qualifies']:
            progress('exact_full_head_TRAIN192',status='rejected');return
        progress('fresh_inner48_outer30');execute('holdouts')
        final=load(OUT/'holdouts/progress.json')
        assert {p:sha(Path(p)) for p in pins}==pins
        progress('finished',status='complete',acceptance=final,no_automatic_deployment=True)
    except BaseException as error:
        progress('failed',status='failed',error=type(error).__name__+': '+str(error));raise

if __name__=='__main__':main()
