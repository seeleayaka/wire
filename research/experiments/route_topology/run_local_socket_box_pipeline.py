"""One frozen new probe followed by scalar review; no retry/automatic deployment."""
import json
import os
from pathlib import Path
import subprocess
from run_prompt_contrast import digest,save,verify
from bundle_runtime_pins import source_pins

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/local_socket_box_source_20261008'


def main():
    path=OUT/'pipeline_progress.json'
    if path.exists():raise FileExistsError('no restart/retry')
    protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'));verify(protocol['pins']);assert source_pins()==protocol['mainline_pins']
    workers=[('fresh_SAM','E:/PythonProject10/runtime/sam3/.venv/Scripts/python.exe',Path(__file__).with_name('run_local_socket_box.py'),['--protocol',str(OUT/'protocol.json')]),
             ('scalar_review','E:/PythonProject10/.venv/Scripts/python.exe',Path(__file__).with_name('analyze_local_socket_box.py'),[])]
    pins={str(f):digest(f) for f in [Path(__file__),*[w[2] for w in workers]]}
    save(OUT/'pipeline_contract.json',dict(pins=pins,no_retry=True,no_deploy=True))
    try:
        for stage,python,worker,args in workers:
            assert all(digest(f)==s for f,s in pins.items())
            save(path,dict(status='running',stage=stage,pid=os.getpid(),no_retry=True))
            with (OUT/(stage+'.log')).open('x',encoding='utf-8') as stream:
                result=subprocess.run([python,'-B',str(worker),*args],cwd=ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1'),stdout=stream,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError(stage+' failed; preserved log/no retry')
        verify(protocol['pins']);assert source_pins()==protocol['mainline_pins']
        save(path,dict(status='complete',scalar_review='PASS',visual_review='pending',
            report_sha256=digest(OUT/'report.json'),new_confirmed_connections=0,deployed=False))
    except BaseException as error:
        save(path,dict(status='failed',error=str(error),no_retry=True));raise


if __name__=='__main__':main()
