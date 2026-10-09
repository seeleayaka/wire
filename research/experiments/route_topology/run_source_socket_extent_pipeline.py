"""One local fresh source-control inference, gate, and independent replay."""
from pathlib import Path
from datetime import datetime,timezone
import subprocess
import os
import sys
from run_review import save

ROOT=Path(__file__).resolve().parents[2]
FOLDER=Path(__file__).parent


def main():
    output=ROOT/'artifacts/mendeley_socket_extent_source_controls_20261006'
    if (output/'pipeline_progress.json').exists():raise FileExistsError('source pipeline already started')
    steps=[('fresh_SAM','E:/PythonProject10/runtime/sam3/.venv/Scripts/python.exe',FOLDER/'run_mendeley_geometry.py',['--protocol',str(output/'protocol.json')]),
           ('source_gate',sys.executable,FOLDER/'source_socket_extent_gate.py',[]),
           ('independent_source_audit',sys.executable,FOLDER/'audit_source_socket_extent_gate.py',[])]
    completed=[];env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1')
    for name,python,script,args in steps:
        save(output/'pipeline_progress.json',{'status':'running','stage':name,'pid':os.getpid(),'completed':completed,
             'updated_utc':datetime.now(timezone.utc).isoformat()})
        with (output/(name+'.log')).open('x',encoding='utf-8') as log:
            child=subprocess.run([python,'-B',str(script),*args],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        if child.returncode!=0:
            save(output/'pipeline_progress.json',{'status':'failed','stage':name,'completed':completed,'exit_code':child.returncode})
            raise RuntimeError(name+' failed; do not automatically retry')
        completed.append(name)
    save(output/'pipeline_progress.json',{'status':'complete','completed':completed,'visual_review':'pending','E_deployed':False})
    print('Source inference/gate/replay finished; inspect actual gate boolean and images before adoption')


if __name__=='__main__':main()
