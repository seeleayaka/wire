"""Supplemental pre-child SHA snapshot for all192 cached alternative inputs."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,REPO,load,save,sha
from current_port_baseline_audit import BASE
from inspection_agent.paired_native_pose import native_pose_runtime_fingerprint


def main():
    out=ROOT/'artifacts/paired_box_regression_input_readiness_20261004'
    if out.exists():raise FileExistsError('Preserve pre-inference input provenance')
    queue=load(ROOT/'artifacts/paired_box_regression_queue_20261004/progress.json')
    assert queue['status']=='queued','Snapshot must precede child computation'
    pins={}
    for entry in load(BASE/'train/report.json')['cases']:
        p=BASE/'train'/(Path(entry['image']).stem+'_predictions.json')
        pins[str(p)]=sha(p)
        row=load(p)
        assert row['image']==entry['image']
        assert row['alternative']['weight_sha256'] in (
            '9ed5ae77c940a78869e084639c55294b05fc65a78e3723f84af2fdc8189ed824',
            'f519a566d98756d988372c3f24dd827480ea41ac7cdb9fd12a97a426756d55b4',
            '18ab6c4c4327cc0b3d9870f5fe36efc6a93d78a6e341ecae05ae703deb4537dc')
    assert len(pins)==192
    out.mkdir();save(out/'report.json',dict(status='pass',before_child=True,
        alternative_inputs=192,pins=pins,runtime=native_pose_runtime_fingerprint(REPO),
        no_new_training=True,no_deployment=True))
    print(dict(status='pass',alternative_inputs=192,before_child=True),flush=True)


if __name__=='__main__':main()
