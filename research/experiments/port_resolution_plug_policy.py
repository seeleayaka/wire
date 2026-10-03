"""Class-bounded resolution residual: loose plugs only, not empty-jack faults."""
import copy
from port_resolution_support import append_resolution
POLICY_ID='accepted_three_model_preserved_resolution_loose_plug_v1_20261003'
def append_resolution_plugs(teacher,current,alternative):
    raw=append_resolution(teacher,current,alternative)
    output=copy.deepcopy(current);output.update(resolution_additions=[],resolution_fallback_reason=raw['resolution_fallback_reason'])
    for candidate in raw['resolution_additions']:
        if candidate['class_id']!=0:continue
        row=copy.deepcopy(candidate);row.update(evidence_tier='resolution_loose_plug_manual_review',
            resolution_policy_id=POLICY_ID,loose_plug_only=True,automatic_fault_verdict=False)
        output['resolution_additions'].append(row);output['all_predictions'].append(row)
    assert output['all_predictions'][:len(current['all_predictions'])]==current['all_predictions']
    assert len(output['all_predictions'])<=len(output['primary'])+5
    return output
