"""One-shot local training-to-evaluation continuation; never deploys or retrains."""
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/port_feature_adaptation_20261003'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(value):(BASE/'continuation_status.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
def main():
    status=BASE/'continuation_status.json'
    if status.exists():raise FileExistsError('One runner per fresh experiment')
    progress=load(BASE/'full/progress.json');training_pid=progress['pid']
    save(dict(status='waiting_for_own_training',training_pid=training_pid,automatic_deployment=False))
    import psutil
    while not (BASE/'full/report.json').exists():
        progress=load(BASE/'full/progress.json')
        if progress['status']=='failed':save(dict(status='training_failed',error=progress.get('error'),automatic_deployment=False));return
        if not psutil.pid_exists(training_pid):save(dict(status='training_process_missing',training_pid=training_pid,automatic_deployment=False));return
        time.sleep(5)
    full=load(BASE/'full/report.json');assert full['status']=='complete'
    for mode in ('train','inner','outer'):
        save(dict(status='evaluating',mode=mode,training_pid=training_pid,automatic_deployment=False))
        with (BASE/('acceptance_'+mode+'.log')).open('x',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,'-B',str(ROOT/'experiments/evaluate_port_feature_adaptation.py'),'--mode',mode],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:save(dict(status='evaluation_failed',mode=mode,exit_code=result.returncode,automatic_deployment=False));return
        report=load(BASE/'evaluation'/mode/'report.json')
        if not report['qualifies']:save(dict(status='candidate_rejected',mode=mode,summary=report['summary'],lost_current_targets=report['lost_current_targets'],automatic_deployment=False));return
    save(dict(status='source_only_accepted_needs_live_reference_and_gui',automatic_deployment=False,
        next='Keep current production. Audit fresh candidate gains, then source/reference/ROI/registration and GUI safety. No automatic model or calibration swap.'))
if __name__=='__main__':main()
