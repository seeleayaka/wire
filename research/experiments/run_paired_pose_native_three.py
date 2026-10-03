"""Scoped fixed learned head + three distinct checkpoints; no retraining."""
import copy
import sys
from pathlib import Path
sys.dont_write_bytecode=True
import run_paired_pose_native_only as gate
from paired_pose_three_vote import select
from prepare_paired_port_semantics import ROOT,sha
OUT=ROOT/'artifacts/paired_pose_native_three_20261004'
PLAN=ROOT/'artifacts/paired_pose_native_three_preregistration_20261004/PLAN.md'


def main():
    old=gate.OUT,gate.PLAN,gate.select,gate.save
    extra={str(p):sha(p) for p in (Path(__file__),Path(__file__).with_name('paired_pose_three_vote.py'),PLAN,
        ROOT/'artifacts/paired_pose_native_only_20261004/report.json')}
    def save(path,value):
        if Path(path).name in ('report.json','protocol.json') and 'pins' in value:
            value=copy.deepcopy(value);value['pins'].update(extra)
            value.update(minimum_distinct_checkpoint_votes=3,only_new_proposals_stricter=True)
        old[3](path,value)
    try:
        gate.OUT,gate.PLAN,gate.select,gate.save=OUT,PLAN,select,save
        gate.main()
        assert {p:sha(Path(p)) for p in extra}==extra
    finally:gate.OUT,gate.PLAN,gate.select,gate.save=old


if __name__=='__main__':main()
