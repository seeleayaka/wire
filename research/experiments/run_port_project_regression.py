"""Whole repository unit suite after optional port integration; isolated output."""
import json,os,sys,time,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];REPO=Path('E:/PythonProject10');OUT=ROOT/'artifacts/port_project_regression_20261003_v3'
sys.dont_write_bytecode=True;sys.path[:0]=[str(REPO),str(REPO/'prototype')]
os.environ.update(QT_QPA_PLATFORM='offscreen',HF_HUB_OFFLINE='1',YOLO_OFFLINE='True',YOLO_AUTOINSTALL='False',YOLO_CONFIG_DIR=str(OUT/'config'))
def main():
    if OUT.exists():raise FileExistsError('Fresh output required')
    (OUT/'config/Ultralytics').mkdir(parents=True)
    import torch;torch.set_num_threads(1)
    started=time.monotonic();suite=unittest.TestLoader().discover(str(REPO/'tests'),pattern='test_*.py')
    with (OUT/'tests.log').open('x',encoding='utf-8') as log:result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    report=dict(status='complete' if result.wasSuccessful() else 'failed',tests=result.testsRun,
        errors=[dict(test=str(t),traceback=trace) for t,trace in result.errors],failures=[dict(test=str(t),traceback=trace) for t,trace in result.failures],
        skipped=[dict(test=str(t),reason=reason) for t,reason in result.skipped],seconds=round(time.monotonic()-started,2),
        unit_tests_only=True,new_model_inference=False,field_acceptance=False)
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report),flush=True)
    if not result.wasSuccessful():raise SystemExit(1)
if __name__=='__main__':main()
