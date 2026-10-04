"""Bounded sequential launch; previous validation success takes priority."""
import os
import subprocess
import sys
import time
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,REPO,load,save,sha
from inspection_agent.paired_native_pose import native_pose_runtime_fingerprint
OUT=ROOT/'artifacts/two_vote_resolution_queue_20261004'
PREVIOUS=ROOT/'artifacts/fine_consensus_rank_holdouts_20261004/report.json'
TARGET=ROOT/'artifacts/two_vote_resolution_20261004'


def main():
    import psutil
    if (OUT/'report.json').exists() or (OUT/'progress.json').exists() or TARGET.exists():raise FileExistsError('Preserve queue/avoid duplicate trial')
    OUT.mkdir(exist_ok=True);script=ROOT/'experiments/run_two_vote_resolution.py';policy=OUT/'POLICY.md'
    preparation=ROOT/'artifacts/two_vote_resolution_preparation_20261004/report.json';assert load(preparation)['status']=='complete'
    paths=[Path(__file__),script,policy,preparation,ROOT/'artifacts/two_vote_resolution_preregistration_20261004/PLAN.md']
    paths.extend(ROOT/'experiments'/n for n in ('unresolved_voter_seeds.py','raw_pose_consensus.py','fine_voter_quality.py','consensus_rank.py','novel_box_geometry.py','relative_port_box.py'))
    pins={str(p):sha(p) for p in paths};frozen=native_pose_runtime_fingerprint(REPO);assert frozen==load(preparation)['runtime']
    save(OUT/'protocol.json',dict(pins=pins,runtime=frozen,wait_limit_seconds=21600,minimum_available_GiB=6,previous_pass_defers_research=True,no_deployment=True))
    start=time.monotonic()
    while True:
        elapsed=time.monotonic()-start
        if elapsed>=21600:raise TimeoutError('Bounded queue expired; no child launched')
        assert all(sha(Path(p))==v for p,v in pins.items()) and native_pose_runtime_fingerprint(REPO)==frozen
        available=psutil.virtual_memory().available/2**30;previous=load(PREVIOUS) if PREVIOUS.exists() else None
        if previous and previous['status']=='holdout_pass_requires_actual_reference_ROI_and_Qt_SAM':
            result=dict(status='deferred_for_previous_actual_acceptance',seconds=round(elapsed,2),no_child_launched=True)
            save(OUT/'report.json',result);save(OUT/'progress.json',result);print(result,flush=True);return
        if previous and previous['status']=='rejected' and available>=6:break
        save(OUT/'progress.json',dict(status='queued',pid=os.getpid(),seconds=round(elapsed,2),available_GiB=round(available,2),previous_finished=bool(previous)))
        time.sleep(10)
    assert not TARGET.exists()
    save(OUT/'progress.json',dict(status='launching',pid=os.getpid(),seconds=round(time.monotonic()-start,2),available_GiB=round(available,2)))
    code=subprocess.call([str(REPO/'.venv/Scripts/python.exe'),'-B',str(script)],cwd=str(ROOT))
    result=dict(status='child_complete' if code==0 else 'child_failed',child_exit_code=code,seconds=round(time.monotonic()-start,2),child_report=str(TARGET/'report.json'))
    save(OUT/'report.json',result);save(OUT/'progress.json',result);print(result,flush=True)


if __name__=='__main__':main()
