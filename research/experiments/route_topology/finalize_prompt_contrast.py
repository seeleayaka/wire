"""Combine final numerical and explicitly qualitative evidence, preserve both."""
import json
import sys
from pathlib import Path
from datetime import datetime,timezone
from analyze_prompt_contrast import OUT
from run_prompt_contrast import save,digest,verify
from run_audit import source_pins


def main():
    protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    measured=json.loads((OUT/'report.json').read_text(encoding='utf-8'))
    backend=json.loads((OUT/'actual_pair_backend/report.json').read_text(encoding='utf-8'))
    verify(protocol['pins'])
    assert source_pins()==protocol['mainline_pins']
    assert measured['status']=='complete' and backend['status']=='complete'
    assert all(case['fresh_cable_exact_reproduction'] for case in measured['cases'])
    assert [case['fresh_cable']['eligible_count'] for case in measured['cases']]==[0,0,13]
    assert [case['fresh_wire']['eligible_count'] for case in measured['cases']]==[0,0,9]
    evidence=[OUT/'protocol.json',OUT/'inference_report.json',OUT/'report.json',
              OUT/'MANUAL_SEMANTICS.zh-CN.md',OUT/'actual_pair_backend/report.json',
              OUT.parent/'route_endpoint_context_20261005/MANUAL_REVIEW.zh-CN.md',
              OUT.parent/'route_topology_20261005_v5/report.json',
              OUT.parent/'route_topology_20261005_v5/tests.json']
    for case in protocol['cases']:
        for prompt in protocol['prompts']:
            run=OUT/case['id']/prompt
            evidence.extend([run/'run_manifest.json',run/'sam/report.json'])
    output=OUT/'review_report.json'
    if output.exists():
        raise FileExistsError('preserve review report')
    save(output,{'status':'complete','created_at':datetime.now(timezone.utc).isoformat(),
         'inference_status':'complete','experiment_decision':'reject_wire_as_replacement',
         'qualitative_model_assisted_review_completed':True,
         'all_new_instance_masks_viewed':221,'authentic_human_terminal_review':False,
         'fresh_image_encoders':3,'prompt_decoders':6,'distinct_cabinets':2,
         'tests_passed':94,'geometry_evidence_counts':{'cable':[0,0,13],'wire':[0,0,9]},
         'new_geometrically_eligible_masks':0,'previous_eligible_masks_lost_under_new_prompt':4,
         'new_confirmed_real_connections':0,'real_automatic_topology_goal_achieved':False,
         'physical_fault_false_positive_rate':None,'field_accuracy':None,
         'real_pair_backend_decisions':[c['decision'] for c in backend['cases']],
         'same_model_observer_count':1,'E_deployed':False,'mainline_unchanged':True,
         'evidence_pins':{str(p.resolve()):digest(p) for p in evidence},
         'next_safe_entry':'Keep original cable mainline. Prioritize verified terminal-identity/port attachment and authentic expected facts; more generic prompt synonyms cannot recover hidden electrical connections.'})
    print(json.dumps({'status':'complete','decision':'reject_wire_as_replacement',
          'new_confirmed_connections':0,'mainline_unchanged':True},ensure_ascii=False))


if __name__=='__main__':
    sys.dont_write_bytecode=True
    main()
