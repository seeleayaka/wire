"""Student proposals, fresh two-context teacher confirmation, no box averaging."""
import copy
from teacher_student_port_policy import merge,valid
from core_port_precision_policy import iou

POLICY=dict(teacher_each_view_score=.75,teacher_seed_iou=.5,teacher_pair_iou=.5,
    maximum_proposals=6,maximum_primary=5,maximum_supplementary=5,
    preserve_accepted_cross_model_cues=True,teacher_views_distinct=True,
    note='Same-photo correlated visual cues; not physical fault evidence')

def proposals(teacher,student):
    accepted=merge(teacher,student,'cross_model_supported')
    consistent=merge(teacher,student,'student_consistent')
    if accepted['fallback_reason'] or consistent['fallback_reason']:return []
    return [copy.deepcopy(p) for p in consistent['student_additions'] if not any(
        iou(p['box_xyxy'],q['box_xyxy'])>=.5 for q in accepted['all_predictions'])][:6]

def confirm_entries(entries):
    result=[]
    for entry in entries:
        windows=entry.get('windows',[]);views=entry.get('views',[]);seed=entry.get('proposal',{})
        if len(windows)!=2 or windows[0]==windows[1] or len(views)!=2 or not valid(seed):continue
        eligible=[[p for p in view if valid(p) and p['class_id']==seed['class_id'] and p['confidence']>.75
            and iou(p['box_xyxy'],seed['box_xyxy'])>=.5] for view in views]
        pairs=[(min(a['confidence'],b['confidence']),a,b) for a in eligible[0] for b in eligible[1]
            if iou(a['box_xyxy'],b['box_xyxy'])>=.5]
        if not pairs:continue
        _,a,b=max(pairs,key=lambda p:p[0]);row=copy.deepcopy(seed)
        row.update(evidence_tier='student_proposal_teacher_two_context_manual_review',
            teacher_context_view_scores=[a['confidence'],b['confidence']],teacher_context_windows=copy.deepcopy(windows))
        result.append(row)
    return sorted(result,key=lambda p:(-p['confidence'],*p['box_xyxy'],p['class_id']))

def extend(teacher,student,entries):
    result=merge(teacher,student,'cross_model_supported');result['teacher_context_additions']=[]
    allowed=proposals(teacher,student)
    remaining=10-len(result['all_predictions'])-(5-len(result['primary']))
    for row in confirm_entries(entries):
        if not any(row['class_id']==p['class_id'] and row['box_xyxy']==p['box_xyxy'] and row['confidence']==p['confidence'] for p in allowed):continue
        if len(result['teacher_context_additions'])>=remaining:break
        if any(iou(row['box_xyxy'],p['box_xyxy'])>=.5 for p in result['all_predictions']):continue
        result['teacher_context_additions'].append(row);result['all_predictions'].append(row)
    return result
