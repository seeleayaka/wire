"""Adaptive same-scale confirmations for missing small cues, at unchanged budget."""
import copy,math
from core_port_context_policy import POLICY as CONTEXT_POLICY,windows,select_zoom,confirmed
from core_port_zoom_policy import ordered
from core_port_supplement_policy import supplement,complete
from core_port_precision_policy import iou
POLICY=copy.deepcopy(CONTEXT_POLICY)
POLICY.update(proposal_maximum_score=1.,rationale='Confirm small cues lacking multi-tile support as well as low-score cues')

def proposals(raw):
    existing=supplement(raw)['all_predictions'];result=[]
    for p in ordered(raw['merged_predictions']):
        l,t,r,b=p['box_xyxy']
        if not all(math.isfinite(v) for v in (*p['box_xyxy'],p['confidence'])):continue
        if not .001<=p['confidence']<=1 or p['class_id'] not in (0,1):continue
        if not 0<min(r-l,b-t)<=max(r-l,b-t)<=160 or not complete(p,raw['source_shape']):continue
        if any(iou(p['box_xyxy'],q['box_xyxy'])>=.5 for q in existing+result):continue
        result.append(copy.deepcopy(p))
        if len(result)==POLICY['maximum_proposals']:break
    return result
