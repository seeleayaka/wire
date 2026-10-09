"""One nonrecurring local job: fresh SAM then analysis, no autonomous retries."""
import json
import os
import subprocess
import sys
from pathlib import Path
from run_prompt_contrast import save,digest
ROOT=Path(__file__).resolve().parents[2]
def main():
    out=ROOT/'artifacts/mendeley_semantic_visible_bundle_20261006'
    if (out/'pipeline_progress.json').exists():raise FileExistsError('pipeline already started')
    folder=Path(__file__).parent
    code_pins={str(p.resolve()):digest(p) for p in [Path(__file__),folder/'analyze_semantic_visible_bundle.py']}
    save(out/'pipeline_contract.json',dict(code_pins=code_pins,
        steps=['fresh_SAM','native_bundle_analysis'],no_automatic_retry=True,
        no_deployment=True,actual_visual_acceptance_pending=True))
    steps=[('fresh_SAM','E:/PythonProject10/runtime/sam3/.venv/Scripts/python.exe',
        folder/'run_mendeley_geometry.py',['--protocol',str(out/'protocol.json')]),
        ('native_bundle_analysis',sys.executable,folder/'analyze_semantic_visible_bundle.py',[])]
    completed=[]
    for name,python,script,args in steps:
        save(out/'pipeline_progress.json',dict(status='running',stage=name,pid=os.getpid(),completed=completed))
        with (out/(name+'.log')).open('x',encoding='utf-8') as stream:
            result=subprocess.run([python,'-B',str(script),*args],cwd=ROOT,
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1'),stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:
            save(out/'pipeline_progress.json',dict(status='failed',stage=name,exit_code=result.returncode,completed=completed))
            raise RuntimeError(name+' failed; preserve evidence, no retry')
        completed.append(name)
    assert all(digest(p)==d for p,d in code_pins.items())
    save(out/'pipeline_progress.json',dict(status='complete',completed=completed,
        actual_visual_review='pending',deployed=False))
    print('Fresh inference and native analysis complete; visual acceptance pending',flush=True)
if __name__=='__main__':main()
