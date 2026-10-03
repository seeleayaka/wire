"""GT-free append-only paired-probability selection, old shared budget."""
import copy
import math
from inspection_agent.port_tiling import box_iou
from inspection_agent.teacher_student_port_support import valid
from port_semantic_verifier import PROBABILITY_GATE


def select(current,proposals,probabilities,head_sha):
    if len(proposals)!=len(probabilities):raise ValueError('probability/proposal mismatch')
    remaining=5-(len(current['all_predictions'])-len(current['primary']))
    if len(current['primary'])>5 or remaining<0:raise ValueError('invalid current budget')
    choices=[]
    for row,prob in zip(proposals,probabilities):
        if len(prob)!=3 or not all(math.isfinite(float(v)) and 0<=float(v)<=1 for v in prob) or abs(sum(prob)-1)>1e-5:
            raise ValueError('invalid classifier probability')
        cls=max(range(3),key=lambda c:prob[c])
        if cls==0 or prob[cls]<PROBABILITY_GATE or cls!=row['class_id']+1:continue
        if not valid(row):raise ValueError('invalid native proposal')
        if len(set(row.get('semantic_model_vote_sha256',[])))<2:raise ValueError('distinct proposal votes missing')
        added=copy.deepcopy(row);added.update(proposal_detector_score=row['confidence'],confidence=float(prob[cls]),
            paired_semantic_probability=float(prob[cls]),paired_head_sha256=head_sha,
            evidence_tier='observed_expected_port_semantics_experimental_manual_review',automatic_fault_verdict=False)
        choices.append(added)
    choices.sort(key=lambda p:(-p['confidence'],*p['box_xyxy'],p['class_id']))
    out=copy.deepcopy(current);out['paired_semantic_additions']=[]
    for row in choices:
        if len(out['paired_semantic_additions'])>=remaining:break
        if any(box_iou(row['box_xyxy'],old['box_xyxy'])>=.5 for old in out['all_predictions']):continue
        out['paired_semantic_additions'].append(row);out['all_predictions'].append(row)
    assert out['all_predictions'][:len(current['all_predictions'])]==current['all_predictions']
    assert len(out['all_predictions'])<=len(current['primary'])+5
    return out
