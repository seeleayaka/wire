"""Finite dependency continuation; no recurring job, deployment, or new training.

Poll only this owned local pipeline. At most180 minutes then record timeout.
The original workflow and checkpoint/script hashes are never modified.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,'E:/PythonProject10')
from inspection_agent.optional_port_crop_review import sha
from inspection_agent.resolution_loose_plug_support import resolution_runtime_fingerprint
ROOT=Path(__file__).resolve().parents[1]
TRAIN=ROOT/'artifacts/port_full_backbone_20261003'
OUT=ROOT/'artifacts/backbone_current_gate_continuation_20261003'


def load(path):return json.loads(path.read_text(encoding='utf-8'))


def save(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');tmp.replace(path)


def main():
    if OUT.exists():raise FileExistsError('No duplicate finite continuation')
    scripts=[Path(__file__),ROOT/'experiments/evaluate_backbone_against_current_policy.py',
             ROOT/'experiments/current_port_baseline_audit.py',ROOT/'experiments/evaluate_full_backbone_acceptance.py']
    pins={str(p):sha(p) for p in scripts}
    frozen=resolution_runtime_fingerprint('E:/PythonProject10')
    OUT.mkdir();started=time.monotonic();last=None
    save(OUT/'protocol.json',dict(pins=pins,current_runtime_fingerprint=frozen,max_wait_minutes=180,
        no_duplicate_training=True,no_original_pipeline_edits=True,no_automatic_deployment=True,
        source_gate_only=True,local_machine_required=True,field_accuracy=False))
    try:
        while time.monotonic()-started<180*60:
            state=load(TRAIN/'pipeline_progress.json')
            signature=(state['status'],state.get('phase'),state.get('pid'))
            if signature!=last:
                save(OUT/'progress.json',dict(status='waiting_owned_pipeline',pid=os.getpid(),owned_pipeline=state,
                     seconds=round(time.monotonic()-started,2)));last=signature
            if state['status']=='failed':
                save(OUT/'progress.json',dict(status='upstream_failed',owned_pipeline=state));return
            if state['status']=='complete':break
            time.sleep(5)
        else:
            save(OUT/'progress.json',dict(status='dependency_timeout',max_wait_minutes=180,no_original_process_terminated=True));return
        assert {p:sha(Path(p)) for p in pins}==pins and resolution_runtime_fingerprint('E:/PythonProject10')==frozen
        with (OUT/'gate.log').open('wb') as log:
            child=subprocess.Popen([sys.executable,'-B',str(scripts[1])],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            save(OUT/'progress.json',dict(status='running_current_baseline_gate',pid=os.getpid(),gate_pid=child.pid))
            code=child.wait()
        if code:
            save(OUT/'progress.json',dict(status='failed',return_code=code,log=str(OUT/'gate.log')));return
        result=load(ROOT/'artifacts/backbone_current_policy_acceptance_20261003/progress.json')
        save(OUT/'progress.json',dict(status='complete',acceptance=result,no_automatic_deployment=True,
             seconds=round(time.monotonic()-started,2)))
    except BaseException as error:
        save(OUT/'progress.json',dict(status='failed',error=type(error).__name__+': '+str(error)));raise


if __name__=='__main__':main()
