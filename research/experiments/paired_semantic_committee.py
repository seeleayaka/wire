"""GT-free unanimity and explicit min-member review score, old shared budget."""
import copy
import math
from inspection_agent.teacher_student_port_support import valid
from inspection_agent.port_tiling import box_iou


def select_committee(current,proposals,member_probabilities,head_shas):
    if len(head_shas)!=3 or len(set(head_shas))!=3 or len(member_probabilities)!=3:
        raise ValueError('three distinct fixed heads required')
    if any(len(member)!=len(proposals) for member in member_probabilities):raise ValueError('candidate count mismatch')
    remaining=5-(len(current['all_predictions'])-len(current['primary']))
    if len(current['primary'])>5 or remaining<0:raise ValueError('current budget invalid')
    choices=[]
    for index,row in enumerate(proposals):
        raw=[member[index] for member in member_probabilities]
        for probability in raw:
            if (len(probability)!=3 or not all(math.isfinite(float(v)) and 0<=v<=1 for v in probability)
                or abs(sum(probability)-1)>1e-5):raise ValueError('invalid member probability')
        cls=row['class_id']+1
        if any(max(range(3),key=lambda i:probability[i])!=cls or probability[cls]<.98 for probability in raw):continue
        if not valid(row) or len(set(row.get('semantic_model_vote_sha256',[])))<2:raise ValueError('invalid native two-model proposal')
        score=min(probability[cls] for probability in raw)
        added=copy.deepcopy(row);added.update(proposal_detector_score=row['confidence'],confidence=score,
            paired_committee_review_score=score,paired_committee_member_probabilities=copy.deepcopy(raw),
            paired_committee_head_sha256=list(head_shas),score_kind='minimum_member_class_probability_not_calibrated_fault_probability',
            evidence_tier='paired_semantic_unanimity_experimental_manual_review',automatic_fault_verdict=False)
        choices.append(added)
    choices.sort(key=lambda row:(-row['confidence'],*row['box_xyxy'],row['class_id']))
    output=copy.deepcopy(current);output['paired_committee_additions']=[]
    for row in choices:
        if len(output['paired_committee_additions'])>=remaining:break
        if any(box_iou(row['box_xyxy'],old['box_xyxy'])>=.5 for old in output['all_predictions']):continue
        output['paired_committee_additions'].append(row);output['all_predictions'].append(row)
    assert output['all_predictions'][:len(current['all_predictions'])]==current['all_predictions']
    return output
