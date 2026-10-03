"""Exclude disallowed additions before FINAL shared-slot allocation only.

Keep the model's native selector unchanged. Filtering its input earlier changes
primary/extra ranking and can lose established plug cues (v2 is rejected).
"""
import copy
from inspection_agent.teacher_student_port_support import complementary_candidates
from inspection_agent.port_tiling import box_iou
POLICY_ID='accepted_three_model_preserved_resolution_loose_plug_v3_20261003'
def append_resolution_plugs_v3(teacher,current,alternative):
    output=copy.deepcopy(current);output.update(resolution_additions=[],resolution_fallback_reason=None)
    candidates=complementary_candidates(teacher,alternative)
    if candidates['fallback_reason']:
        output['resolution_fallback_reason']=candidates['fallback_reason'];return output
    remaining=5-(len(current['all_predictions'])-len(current['primary']))
    for row in candidates['student_additions']:
        if row['class_id']!=0:continue
        if len(output['resolution_additions'])>=remaining:break
        if any(box_iou(row['box_xyxy'],old['box_xyxy'])>=.5 for old in output['all_predictions']):continue
        row=copy.deepcopy(row);row.update(evidence_tier='resolution_loose_plug_manual_review',inference_imgsz=1280,
            resolution_policy_id=POLICY_ID,loose_plug_only=True,automatic_fault_verdict=False,
            allowed_class_filtered_before_final_budget=True)
        output['resolution_additions'].append(row);output['all_predictions'].append(row)
    assert output['all_predictions'][:len(current['all_predictions'])]==current['all_predictions']
    assert len(output['all_predictions'])<=len(output['primary'])+5
    return output
