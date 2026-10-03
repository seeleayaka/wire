"""Uniform poses from model boxes; no GT, photo names or absolute coordinates."""
import copy
import math
from inspection_agent.paired_port_median_features import proposals as median_proposals
from inspection_agent.paired_port_features import select as original_select
from inspection_agent.port_tiling import box_iou


def poses(box):
    l,t,r,b=map(float,box);w,h=r-l,b-t;cx,cy=(l+r)/2,(t+b)/2
    if min(w,h)<=0:raise ValueError('Invalid pose seed')
    result=[list(map(float,box))]
    for dx,dy in ((-.15*w,0),(.15*w,0),(0,-.15*h),(0,.15*h)):result.append([l+dx,t+dy,r+dx,b+dy])
    for scale in (.8,1.2):result.append([cx-w*scale/2,cy-h*scale/2,cx+w*scale/2,cy+h*scale/2])
    return result


def proposals(teacher,models,current):
    h,w=teacher['predictions']['source_shape'];pool=[]
    for model in models:
        if model['source_sha256']!=teacher['source_sha256'] or model['predictions']['source_shape']!=[h,w]:
            raise ValueError('Pose source/model geometry mismatch')
        for row in model['predictions']['merged_predictions']:
            l,t,r,b=row['box_xyxy']
            if row['confidence']>.05 and 16<=l<r<=w-16 and 16<=t<b<=h-16:pool.append((model['weight_sha256'],row))
    result=[];seen=set()
    for parent,seed in enumerate(median_proposals(teacher,models)):
        if any(box_iou(seed['box_xyxy'],old['box_xyxy'])>=.5 for old in current['all_predictions']):continue
        for variant,box in enumerate(poses(seed['box_xyxy'])):
            l,t,r,b=box
            if not (16<=l<r<=w-16 and 16<=t<b<=h-16):continue
            if any(box_iou(box,old['box_xyxy'])>=.5 for old in current['all_predictions']):continue
            votes={weight for weight,row in pool if row['class_id']==seed['class_id'] and box_iou(box,row['box_xyxy'])>=.5}
            key=(seed['class_id'],*map(float,box))
            if len(votes)<2 or key in seen:continue
            seen.add(key);row=copy.deepcopy(seed)
            row.update(box_xyxy=list(map(float,box)),semantic_model_vote_sha256=sorted(votes),
                pose_parent_seed_id=parent,pose_variant_index=variant,pose_seed_box=copy.deepcopy(seed['box_xyxy']))
            result.append(row)
    return result


def select(current,candidates,probabilities,head_sha):
    if len(candidates)!=len(probabilities):raise ValueError('Native probability count mismatch')
    chosen={}
    for row,prob in zip(candidates,probabilities):
        if len(prob)!=3 or not all(math.isfinite(float(v)) and 0<=float(v)<=1 for v in prob) or abs(sum(prob)-1)>1e-5:
            raise ValueError('Invalid native probability')
        cls=max(range(3),key=lambda i:prob[i])
        if len(set(row['semantic_model_vote_sha256']))<3 or cls!=row['class_id']+1 or prob[cls]<.98:continue
        parent=row['pose_parent_seed_id'];rank=(-prob[cls],*row['box_xyxy'],row['class_id'])
        if parent not in chosen or rank<chosen[parent][0]:chosen[parent]=(rank,row,prob)
    rows=[v[1] for v in chosen.values()];scores=[v[2] for v in chosen.values()]
    return original_select(current,rows,scores,head_sha)
