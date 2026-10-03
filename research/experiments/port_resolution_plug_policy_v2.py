"""Apply allowed-class filtering BEFORE allocating the shared candidate budget."""
import copy
from port_resolution_plug_policy import append_resolution_plugs
POLICY_ID='accepted_three_model_preserved_resolution_loose_plug_v2_20261003'
def append_resolution_plugs_v2(teacher,current,alternative):
    filtered=copy.deepcopy(alternative)
    for key in ('merged_predictions','edge_kept_predictions'):
        filtered['predictions'][key]=[row for row in filtered['predictions'][key] if row['class_id']==0]
    filtered['zoom_evidence']=[entry for entry in filtered['zoom_evidence'] if entry['proposal']['class_id']==0]
    result=append_resolution_plugs(teacher,current,filtered)
    for row in result['resolution_additions']:
        row.update(resolution_policy_id=POLICY_ID,allowed_class_filtered_before_budget=True)
    return result
