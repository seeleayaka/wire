"""Experimental independent semantic confirmation, never replace old cues."""
import copy,math
from core_port_precision_policy import iou
from port_semantic_verifier import PROBABILITY_GATE

def complete(row,shape):
    h,w=shape;l,t,r,b=row['box_xyxy']
    return 16<=l<r<=w-16 and 16<=t<b<=h-16

def proposals(teacher,models):
    shape=teacher['predictions']['source_shape'];rows=[]
    for model in models:
        if model['source_sha256']!=teacher['source_sha256'] or model['predictions']['source_shape']!=shape:
            raise ValueError('semantic_source_geometry_mismatch')
        raw=model['predictions']['edge_kept_predictions']
        for candidate in model['predictions']['merged_predictions']:
            if candidate['confidence']<=.25 or not complete(candidate,shape):continue
            if not any(t['class_id']==candidate['class_id'] and t['confidence']>.25 and
                iou(t['box_xyxy'],candidate['box_xyxy'])>=.5 for t in teacher['predictions']['merged_predictions']):continue
            tiles={r['source_tile'] for r in raw if r['class_id']==candidate['class_id'] and r['confidence']>.25 and
                complete(r,shape) and iou(r['box_xyxy'],candidate['box_xyxy'])>=.5}
            if len(tiles)<2:continue
            row=copy.deepcopy(candidate);row.update(semantic_detector_weight_sha256=model['weight_sha256'],
                semantic_source_votes=len(tiles));rows.append(row)
    selected=[]
    for row in sorted(rows,key=lambda r:-r['confidence']):
        if not any(iou(row['box_xyxy'],old['box_xyxy'])>=.5 for old in selected):selected.append(row)
    return selected

def append_semantic(current,candidates,probabilities):
    if len(candidates)!=len(probabilities):raise ValueError('semantic_probability_count_mismatch')
    output=copy.deepcopy(current);output['semantic_additions']=[]
    remaining=5-(len(current['all_predictions'])-len(current['primary']))
    for candidate,probability in zip(candidates,probabilities):
        if len(output['semantic_additions'])>=remaining:break
        if len(probability)!=3 or not all(math.isfinite(p) and 0<=p<=1 for p in probability) or abs(sum(probability)-1)>1e-5:
            raise ValueError('invalid_semantic_probabilities')
        if candidate['class_id'] not in (0,1):raise ValueError('invalid_semantic_candidate_class')
        cls=candidate['class_id']+1
        if probability[cls]<PROBABILITY_GATE or max(range(3),key=lambda k:probability[k])!=cls:continue
        if any(iou(candidate['box_xyxy'],old['box_xyxy'])>=.5 for old in output['all_predictions']):continue
        row=copy.deepcopy(candidate);row.update(evidence_tier='semantic_supported_manual_review',
            semantic_probability=float(probability[cls]),semantic_threshold=PROBABILITY_GATE)
        output['semantic_additions'].append(row);output['all_predictions'].append(row)
    assert output['all_predictions'][:len(current['all_predictions'])]==current['all_predictions']
    assert len(output['all_predictions'])<=len(output['primary'])+5 and len(output['primary'])<=5
    return output
