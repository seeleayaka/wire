"""Two correlated views must agree; min score is explicitly uncalibrated."""
import copy,math
from inspection_agent.teacher_student_port_support import valid
from inspection_agent.port_tiling import box_iou


def select(current,proposals,members,head_shas):
    if len(head_shas)!=2 or len(set(head_shas))!=2 or len(members)!=2 or any(len(m)!=len(proposals) for m in members):
        raise ValueError('Two distinct heads and matched proposals required')
    remaining=5-(len(current['all_predictions'])-len(current['primary']))
    if len(current['primary'])>5 or remaining<0:raise ValueError('Invalid current budget')
    choices=[]
    for index,row in enumerate(proposals):
        raw=[m[index] for m in members]
        for p in raw:
            if len(p)!=3 or not all(math.isfinite(float(v)) and 0<=v<=1 for v in p) or abs(sum(p)-1)>1e-5:raise ValueError('Invalid view probabilities')
        cls=row['class_id']+1
        if any(max(range(3),key=lambda i:p[i])!=cls or p[cls]<.98 for p in raw):continue
        if not valid(row) or len(set(row.get('semantic_model_vote_sha256',[])))<2:raise ValueError('Invalid two-checkpoint native witness')
        score=min(p[cls] for p in raw);added=copy.deepcopy(row)
        added.update(proposal_detector_score=row['confidence'],confidence=score,paired_crossview_review_score=score,
            paired_crossview_member_probabilities=copy.deepcopy(raw),paired_crossview_head_sha256=list(head_shas),
            score_kind='minimum_correlated_member_probability_not_calibrated_fault_probability',
            evidence_tier='context_footprint_agreement_experimental_manual_review',automatic_fault_verdict=False)
        choices.append(added)
    choices.sort(key=lambda row:(-row['confidence'],*row['box_xyxy'],row['class_id']))
    output=copy.deepcopy(current);output['paired_crossview_additions']=[]
    for row in choices:
        if len(output['paired_crossview_additions'])>=remaining:break
        if any(box_iou(row['box_xyxy'],old['box_xyxy'])>=.5 for old in output['all_predictions']):continue
        output['paired_crossview_additions'].append(row);output['all_predictions'].append(row)
    assert output['all_predictions'][:len(current['all_predictions'])]==current['all_predictions']
    return output
