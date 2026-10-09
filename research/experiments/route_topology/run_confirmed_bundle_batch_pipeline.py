"""Local non-recurring frozen batch: inference -> analysis -> independent audit."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from run_review import save

ROOT=Path(__file__).resolve().parents[2]
FOLDER=Path(__file__).parent


def main():
    output=ROOT/'artifacts/mendeley_confirmed_bundle_batch_v2_20261006'
    if (output/'pipeline_progress.json').exists():raise FileExistsError('pipeline already started')
    protocol=output/'protocol.json'
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1')
    steps=[('fresh_SAM','E:/PythonProject10/runtime/sam3/.venv/Scripts/python.exe',
            FOLDER/'run_bundle_batch_geometry_v2.py',['--protocol',str(protocol)]),
           ('analysis',sys.executable,FOLDER/'analyze_confirmed_bundle_batch_v2.py',[]),
           ('independent_audit',sys.executable,FOLDER/'audit_confirmed_bundle_batch_v2.py',[])]
    completed=[]
    for name,python,script,args in steps:
        save(output/'pipeline_progress.json',{'status':'running','stage':name,'pid':os.getpid(),
            'completed':completed,'updated_utc':datetime.now(timezone.utc).isoformat()})
        with (output/(name+'.log')).open('x',encoding='utf-8') as stream:
            result=subprocess.run([python,'-B',str(script),*args],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode!=0:
            save(output/'pipeline_progress.json',{'status':'failed','stage':name,'exit_code':result.returncode,
                'completed':completed,'log':str(output/(name+'.log'))})
            raise RuntimeError(name+' failed; retain evidence, do not retry automatically')
        completed.append(name)
    save(output/'pipeline_progress.json',{'status':'complete','completed':completed,
        'actual_visual_review':'pending','deployed':False,'electrical_correctness':'not_assessed',
        'not_blind_or_field_accuracy':True,'updated_utc':datetime.now(timezone.utc).isoformat()})
    print('Complete: full-batch inference/analysis/audit; actual visual review still pending',flush=True)


if __name__=='__main__':main()
