"""Finite cached-semantics/RGB head training, same feasibility/source gates."""
import json
import subprocess
import sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/dense_pixel_port_probe_20261003'
sys.path.insert(0,'E:/PythonProject10')
from inspection_agent.optional_port_crop_review import sha


def load(path):return json.loads(path.read_text(encoding='utf-8'))


def save(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');tmp.replace(path)


def main():
    if (BASE/'pipeline_progress.json').exists():raise FileExistsError('No duplicate pipeline')
    smoke=load(BASE/'smoke/report.json')
    assert smoke['status']=='complete' and smoke['changed_pixel_tensors']>0 and smoke['frozen_encoder_unchanged']
    scripts=[ROOT/'experiments/train_dense_pixel_port_probe.py',ROOT/'experiments/evaluate_dense_pixel_source_support.py',
        ROOT/'experiments/dense_pixel_port_probe.py',ROOT/'experiments/dense_port_probe.py',
        ROOT/'experiments/evaluate_dense_dino_source_support.py',ROOT/'experiments/current_port_baseline_audit.py',Path(__file__)]
    pins={str(p):sha(p) for p in scripts}
    save(BASE/'pipeline_protocol.json',dict(pins=pins,frozen_verified_DINO_features_reused=True,
        pixel_branch_updates_proved=True,fixed_head_epochs=8,crop_precision90_recall25_required=True,
        same_scores_iou_and_budget=True,train192_then_inner48_then_outer30=True,baseline='Current V3 277/63/33',
        coarse_failure_preserved=True,no_automatic_deployment=True,local_machine_required=True,field_accuracy=False))
    for phase,index,args in (('fine_head_training',0,['--mode','full']),('source_acceptance',1,[])):
        assert {p:sha(Path(p)) for p in pins}==pins
        with (BASE/(phase+'.log')).open('wb') as log:
            child=subprocess.Popen([sys.executable,'-B',str(scripts[index]),*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            save(BASE/'pipeline_progress.json',dict(status='running',phase=phase,pid=child.pid))
            code=child.wait()
        if code:save(BASE/'pipeline_progress.json',dict(status='failed',phase=phase,return_code=code));return
        if phase=='fine_head_training' and not load(BASE/'full/report.json')['qualifies_crop_feasibility']:
            save(BASE/'pipeline_progress.json',dict(status='rejected_crop_feasibility',
                 crop_feasibility=load(BASE/'full/report.json')['crop_feasibility'],no_source_accuracy_evaluated=True));return
    save(BASE/'pipeline_progress.json',dict(status='complete',acceptance=load(BASE/'source_acceptance/progress.json'),no_automatic_deployment=True))


if __name__=='__main__':main()
