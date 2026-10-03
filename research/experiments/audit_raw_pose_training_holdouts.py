"""Reuse independent bounded audit against the separately frozen trained head."""
from prepare_paired_port_semantics import ROOT,load,sha
import audit_raw_pose_consensus as audit

def main():
    audit.SOURCE=ROOT/'artifacts/raw_pose_training_holdouts_20261004'
    audit.OUT=ROOT/'artifacts/raw_pose_training_holdouts_audit_20261004'
    head=ROOT/'artifacts/raw_pose_training_20261004/full/last_head.pt'
    report=load(audit.SOURCE/'report.json')
    assert sha(head)==report['pins'][str(head)]
    audit.HEAD_SHA=sha(head);audit.main()

if __name__=='__main__':main()
