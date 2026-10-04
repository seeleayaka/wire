"""Two-checkpoint seeds for an evidence-gathering action, never final cues."""
import copy
import math
from inspection_agent.paired_native_pose_features import poses
from inspection_agent.port_tiling import box_iou
from novel_box_geometry import is_duplicate


def pool_rows(models,source_sha,shape):
    h,w=shape;pool=[]
    for model in models:
        if model['source_sha256']!=source_sha or model['predictions']['source_shape']!=list(shape):raise ValueError('Unresolved seed source/frame mismatch')
        for row in model['predictions']['merged_predictions']:
            l,t,r,b=map(float,row['box_xyxy'])
            if not all(math.isfinite(v) for v in (l,t,r,b)):raise ValueError('Nonfinite unresolved model geometry')
            if row['confidence']>.05 and 16<=l<r<=w-16 and 16<=t<b<=h-16:pool.append((model['weight_sha256'],row))
    return pool


def seeds(models,current,source_sha,shape):
    h,w=shape;pool=pool_rows(models,source_sha,shape)
    pool.sort(key=lambda p:(-p[1]['confidence'],p[1]['class_id'],*p[1]['box_xyxy'],p[0]))
    parents=[];seen=set();result=[]
    for weight,seed in pool:
        parent=next((i for i,r in enumerate(parents) if r['class_id']==seed['class_id'] and box_iou(r['box_xyxy'],seed['box_xyxy'])>=.5),None)
        if parent is None:parent=len(parents);parents.append(seed)
        for variant,box in enumerate(poses(seed['box_xyxy'])):
            l,t,r,b=box
            if not (16<=l<r<=w-16 and 16<=t<b<=h-16) or is_duplicate(box,current['all_predictions']):continue
            best={}
            for digest,row in pool:
                if row['class_id']!=seed['class_id']:continue
                value=box_iou(box,row['box_xyxy'])
                if value>=.5:best[digest]=max(value,best.get(digest,0.))
            key=(seed['class_id'],*box)
            if len(best)!=2 or key in seen:continue
            seen.add(key);row=copy.deepcopy(seed)
            row.update(box_xyxy=box,semantic_model_vote_sha256=sorted(best),localization_voter_best_IoU=best,
                pose_parent_seed_id=parent,pose_variant_index=variant,pose_seed_box=copy.deepcopy(seed['box_xyxy']),
                seed_only_requires_third_checkpoint=True,automatic_fault_verdict=False)
            result.append(row)
    return result


def windows(seed,shape):
    h,w=shape
    if min(h,w)<960:raise ValueError('Unsupported local evidence view size')
    l,t,r,b=seed['box_xyxy'];cx,cy=(l+r)/2,(t+b)/2
    return [[x,y,x+960,y+960] for dx,dy in ((-120,-120),(120,120))
        for x,y in [(max(0,min(w-960,round(cx+dx-480))),max(0,min(h-960,round(cy+dy-480))))]]
