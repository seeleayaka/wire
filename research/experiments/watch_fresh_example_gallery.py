"""Finite local gallery builder for an already authorized four-case run."""
import json
import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/fresh_four_examples_20261004'


def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args()
    OUT=args.output.resolve()
    log=OUT/'gallery_watch.json'
    if log.exists():
        raise FileExistsError('one gallery watcher only')
    start=time.monotonic()
    count=-1
    while time.monotonic()-start<12*3600:
        try:
            progress=json.loads((OUT/'progress.json').read_text(encoding='utf-8'))
        except (OSError,json.JSONDecodeError):
            time.sleep(10)
            continue
        now=len(progress.get('cases',[]))
        if now!=count or progress['status']!='running':
            rendered=subprocess.run([sys.executable,'-B',str(ROOT/'experiments/present_fresh_examples.py'),'--output',str(OUT)],
                cwd=str(ROOT),capture_output=True,text=True,encoding='utf-8',errors='replace')
            result=dict(status=progress['status'],completed=now,render_returncode=rendered.returncode,
                seconds=round(time.monotonic()-start,2),gallery=str(OUT/'index.html'))
            if rendered.returncode:
                result['render_error']=rendered.stderr[-1000:]
            log.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
            print(json.dumps(result),flush=True)
            if rendered.returncode:
                return
            audit_args=[sys.executable,'-B',str(ROOT/'experiments/audit_fresh_examples.py'),'--output',str(OUT)]
            if progress['status']=='complete':
                audit_args.append('--final')
            audited=subprocess.run(audit_args,cwd=str(ROOT),capture_output=True,text=True,
                encoding='utf-8',errors='replace')
            result['freshness_audit_returncode']=audited.returncode
            if audited.returncode:
                result['audit_error']=audited.stderr[-1500:]
            log.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
            count=now
        if progress['status']!='running':
            return
        time.sleep(15)
    log.write_text(json.dumps(dict(status='watch_timeout',completed=count,
        inference_cancelled=False,seconds=round(time.monotonic()-start,2)),indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
