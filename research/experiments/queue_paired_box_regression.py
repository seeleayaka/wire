"""Bounded sequencing: no fourth competing model process at low memory."""
import os
import subprocess
import sys
import time
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,REPO,load,save,sha
from inspection_agent.paired_native_pose import native_pose_runtime_fingerprint
OUT=ROOT/'artifacts/paired_box_regression_queue_20261004'
SCRIPT=Path(__file__).with_name('run_paired_box_regression_geometry.py')
UPSTREAM=ROOT/'artifacts/photometric_native_proposals_20261004/full'


def main():
    import psutil
    if OUT.exists():raise FileExistsError('Preserve bounded queue evidence')
    OUT.mkdir();started=time.monotonic();frozen=native_pose_runtime_fingerprint(REPO)
    pins={str(p):sha(p) for p in (Path(__file__),SCRIPT,Path(__file__).with_name('relative_port_box.py'),
        ROOT/'artifacts/paired_box_regression_preregistration_20261004/PLAN.md')}
    save(OUT/'protocol.json',dict(pins=pins,runtime=frozen,maximum_wait_seconds=21600,
        no_photometric_or_runtime_changes=True,fixed_next_script=str(SCRIPT)))
    try:
        while True:
            assert all(sha(Path(p))==v for p,v in pins.items())
            assert native_pose_runtime_fingerprint(REPO)==frozen
            status=load(UPSTREAM/'progress.json')['status']
            if status=='failed':raise RuntimeError('Prior finite job failed; do not restart or bypass')
            if status in ('rejected','source_pass_requires_actual_reference_ROI_gates') and psutil.virtual_memory().available>6*2**30:break
            if time.monotonic()-started>21600:raise TimeoutError('Bounded sequencing wait expired')
            save(OUT/'progress.json',dict(status='queued',pid=os.getpid(),upstream_status=status,
                reason='await_prior_finite_proposal_job_and_safe_memory',wait_seconds=round(time.monotonic()-started,2)))
            time.sleep(30)
        save(OUT/'progress.json',dict(status='running_child',pid=os.getpid(),script=str(SCRIPT)))
        subprocess.run([sys.executable,str(SCRIPT)],cwd=str(ROOT),check=True)
        assert all(sha(Path(p))==v for p,v in pins.items()) and native_pose_runtime_fingerprint(REPO)==frozen
        result=load(ROOT/'artifacts/paired_box_regression_geometry_20261004/report.json')
        save(OUT/'progress.json',dict(status='complete',child_status=result['status'],summary=result['summary']))
        print(dict(status='complete',child_status=result['status'],summary=result['summary']),flush=True)
    except BaseException as error:
        save(OUT/'progress.json',dict(status='failed',error=type(error).__name__+': '+str(error)));raise


if __name__=='__main__':main()
