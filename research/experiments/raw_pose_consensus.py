"""Raw detector seeds, same poses and strict checkpoint consensus; GT-free."""
import copy
import math
from inspection_agent.paired_native_pose_features import poses,select
from inspection_agent.port_tiling import box_iou

def proposals(teacher,models,current):
    h,w=teacher['predictions']['source_shape'];pool=[]
    for model in models:
        if model['source_sha256']!=teacher['source_sha256'] or model['predictions']['source_shape']!=[h,w]:raise ValueError('raw pose geometry mismatch')
        for row in model['predictions']['merged_predictions']:
            l,t,r,b=map(float,row['box_xyxy'])
            if not all(math.isfinite(v) for v in (l,t,r,b)):raise ValueError('nonfinite raw pose')
            if row['confidence']>.05 and 16<=l<r<=w-16 and 16<=t<b<=h-16:pool.append((model['weight_sha256'],row))
    pool.sort(key=lambda pair:(-pair[1]['confidence'],pair[1]['class_id'],*pair[1]['box_xyxy'],pair[0]))
    parents=[];seen=set();result=[]
    for digest,seed in pool:
        parent=next((i for i,row in enumerate(parents) if row['class_id']==seed['class_id'] and box_iou(row['box_xyxy'],seed['box_xyxy'])>=.5),None)
        if parent is None:parent=len(parents);parents.append(seed)
        for variant,box in enumerate(poses(seed['box_xyxy'])):
            l,t,r,b=box
            if not (16<=l<r<=w-16 and 16<=t<b<=h-16):continue
            if any(box_iou(box,row['box_xyxy'])>=.5 for row in current['all_predictions']):continue
            votes={weight for weight,row in pool if row['class_id']==seed['class_id'] and box_iou(box,row['box_xyxy'])>=.5}
            key=(seed['class_id'],*box)
            if len(votes)<3 or key in seen:continue
            seen.add(key);row=copy.deepcopy(seed)
            row.update(box_xyxy=box,semantic_detector_weight_sha256=digest,semantic_model_vote_sha256=sorted(votes),
                pose_parent_seed_id=parent,pose_variant_index=variant,pose_seed_box=copy.deepcopy(seed['box_xyxy']),raw_seed_geometry=True)
            result.append(row)
    return result
