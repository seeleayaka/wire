"""Explicit architecture controls, preserved gates and accepted baseline."""
import argparse,os,sys
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,load,save,sha
from prepare_boxpool_control import prepare
import run_paired_boxpool_stage as adapter
import evaluate_paired_boxpool_full_train as full_evaluator
from run_paired_shape_nonlinear_probe import fit_head


def main(control):
    out=ROOT/('artifacts/paired_boxpool_original_rows_20261003' if control=='original' else 'artifacts/paired_boxpool_nonlinear_20261003')
    if out.exists():raise FileExistsError('Preserve controlled run')
    prepare(out,original_only=control=='original');pins={str(p):sha(p) for p in (Path(__file__),
        Path(__file__).with_name('prepare_boxpool_control.py'),Path(adapter.__file__),Path(full_evaluator.__file__),
        Path(__file__).with_name('run_paired_shape_nonlinear_probe.py'),ROOT/'artifacts/paired_boxpool_controls_preregistration_20261003/PLAN.md')}
    save(out/'protocol.json',dict(pins=pins,control=control,architecture=[6144,3] if control=='original' else [6144,32,'GELU',3],
        fixed_steps=400,no_reextraction=True,no_deployment=True))
    def progress(phase,**kw):
        record=dict(status='running',pid=os.getpid(),phase=phase);record.update(kw);save(out/'progress.json',record)
    old_out=adapter.OUT
    try:
        adapter.OUT=out
        for stage in ('classifier','source','full'):
            progress(stage)
            module=adapter.importlib.import_module(adapter.MODULES[stage]);old_fit=getattr(module,'fit_head',None)
            try:
                if control=='nonlinear' and old_fit is not None:module.fit_head=fit_head
                adapter.execute(stage)
            finally:
                if old_fit is not None:module.fit_head=old_fit
            report=load(out/('heads_oof/report.json' if stage=='classifier' else 'source_train_oof/report.json' if stage=='source' else 'head_full/report.json'))
            qualifies=report.get('qualifies_crop_feasibility') if stage=='classifier' else report.get('qualifies') if stage=='source' else True
            if not qualifies:
                progress(stage,status='rejected');assert {p:sha(Path(p)) for p in pins}==pins;return
        if control=='nonlinear':
            # Different architecture must get its own full-head factory, never
            # monkey-patch torch globally or load these weights into Linear.
            progress('full_head_created_pending_architecture_specific_source_gate',status='complete',no_deployment=True)
            return
        previous=full_evaluator.OUT
        try:full_evaluator.OUT=out;progress('exact_full_TRAIN192');full_evaluator.main()
        finally:full_evaluator.OUT=previous
        if not load(out/'full_train/report.json')['qualifies']:
            progress('exact_full_TRAIN192',status='rejected');return
        progress('fresh_inner48_outer30');adapter.execute('holdouts')
        assert {p:sha(Path(p)) for p in pins}==pins
        progress('finished',status='complete',acceptance=load(out/'holdouts/progress.json'),no_deployment=True)
    except BaseException as error:
        progress('failed',status='failed',error=type(error).__name__+': '+str(error));raise
    finally:adapter.OUT=old_out

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('control',choices=('original','nonlinear'));main(parser.parse_args().control)
