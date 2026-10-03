"""Exact bounded independent replay of the rejected fine native proposal trial."""
from prepare_paired_port_semantics import ROOT
import audit_raw_pose_consensus as audit

def main():
    audit.SOURCE=ROOT/'artifacts/fine_native_consensus_20261004'
    audit.OUT=ROOT/'artifacts/fine_native_consensus_audit_20261004'
    audit.main()

if __name__=='__main__':main()
