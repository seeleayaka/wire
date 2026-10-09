"""Wait only for this started trial; analyze/audit once, never launch SAM."""
import json
import os
from pathlib import Path
import subprocess
import time
from run_prompt_contrast import digest,save

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/local_socket_crop_source_20261008'


def main():
    path=OUT/'finalization_progress.json'
    if path.exists():raise FileExistsError('one finisher; no automatic retry')
    workers=[Path(__file__).with_name('analyze_local_socket_crop.py'),Path(__file__).with_name('audit_local_socket_crop.py')]
    pins={str(f):digest(f) for f in [Path(__file__),*workers]};start=time.monotonic()
    save(path,dict(status='waiting_for_started_inference',pid=os.getpid(),workers=pins,no_model_launch=True))
    try:
        while not (OUT/'inference_report.json').exists():
            status=json.loads((OUT/'progress.json').read_text(encoding='utf-8'))
            if status['status']=='failed':raise RuntimeError('inference failed; no retry')
            if time.monotonic()-start>1900:raise TimeoutError('wait budget exceeded')
            time.sleep(5)
        for worker in workers:
            assert all(digest(f)==s for f,s in pins.items()),'finisher worker drift'
            save(path,dict(status='running',stage=worker.stem,pid=os.getpid(),workers=pins,no_model_launch=True))
            with (OUT/(worker.stem+'.log')).open('x',encoding='utf-8') as stream:
                result=subprocess.run(['E:/PythonProject10/.venv/Scripts/python.exe','-B',str(worker)],cwd=ROOT,
                    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdout=stream,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError(worker.stem+' failed; preserved log/no retry')
        save(path,dict(status='complete',independent_audit='PASS',visual_review='pending',
            report_sha256=digest(OUT/'report.json'),audit_sha256=digest(OUT/'audit_report.json'),
            workers=pins,no_model_launch=True,deployed=False,new_confirmed_connections=0))
    except BaseException as error:
        save(path,dict(status='failed',error=str(error),no_retry=True,no_model_launch=True));raise


if __name__=='__main__':main()
