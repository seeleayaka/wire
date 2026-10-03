"""Readonly source metrics audit plus explicit terminal-status interpretation."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
from prepare_paired_port_semantics import ROOT,load,save,sha


def main():
    out=ROOT/'artifacts/paired_committee_reference_veto_20261003'
    protocol=load(out/'protocol.json');assert {p:sha(Path(p)) for p in protocol['pins']}==protocol['pins']
    reports={stage:load(out/stage/'report.json') for stage in ('train','inner','outer')}
    assert all(report['status']=='complete' for report in reports.values())
    assert reports['train']['qualifies'] and reports['inner']['qualifies'] and not reports['outer']['qualifies']
    interpretation=dict(status='rejected_source_outer',all270_reference_candidate_veto_scoring_complete=True,
        metrics={stage:report['summary'] for stage,report in reports.items()},
        terminal_error='progress writer supplied status keyword twice after metrics saved and pins verified',
        scientific_rejection_precedes_harness_status_error=True,no_detector_crash=True,no_deployment=True,
        all_original_protocol_pins_verified=True,old_progress_running_is_stale=True,field_accuracy=False)
    save(out/'run_interpretation.json',interpretation);print(str(interpretation))


if __name__=='__main__':main()
