"""Fixed 640px two-context confirmations for novel multiscale-trained student cues."""
import copy
from teacher_context_student_policy import proposals,confirm_entries
from teacher_student_port_policy import merge
from core_port_precision_policy import iou
from inspection_agent.port_tiling import near_artificial_edge

POLICY=dict(crop_size=640,shift=80,predict_imgsz=960,each_view_score=.75,each_seed_iou=.5,pair_iou=.5,
    maximum_proposals=6,maximum_primary=5,maximum_supplementary=5,
    preserve_all_accepted_cross_model_cues=True,independent_physical_evidence=False)

def windows(seed,shape):
    h,w=shape
    if min(h,w)<640:raise ValueError('image_too_small_for_microcontext')
    l,t,r,b=seed['box_xyxy'];cx=(l+r)/2;cy=(t+b)/2;result=[]
    for dx,dy in ((-80,-80),(80,80)):
        x=max(0,min(w-640,round(cx+dx-320)));y=max(0,min(h-640,round(cy+dy-320)))
        result.append([x,y,x+640,y+640])
    return result

def predict_views(model,image,seeds):
    import torch
    h,w=image.shape[:2];entries=[]
    for seed in seeds:
        crop_windows=windows(seed,[h,w]);crops=[image[y:b,x:r] for x,y,r,b in crop_windows]
        outputs=model.predict(crops,imgsz=960,conf=.001,iou=.7,max_det=300,device='cpu',verbose=False,save=False)
        torch.set_num_threads(4)
        if len(outputs)!=2:raise ValueError('microcontext_batch_mismatch')
        views=[]
        for index,(output,window) in enumerate(zip(outputs,crop_windows)):
            x,y,_,_=window;rows=[]
            for box in output.boxes:
                local=list(map(float,box.xyxy[0].tolist()))
                if near_artificial_edge(local,window,w,h,16):continue
                l,t,r,b=local;rows.append(dict(box_xyxy=[l+x,t+y,r+x,b+y],confidence=float(box.conf.item()),class_id=int(box.cls.item()),microcontext_view=index))
            views.append(rows)
        entries.append(dict(proposal=copy.deepcopy(seed),windows=crop_windows,views=views))
    return entries

def extend(teacher,student,entries):
    result=merge(teacher,student,'cross_model_supported');result['microcontext_additions']=[]
    allowed=proposals(teacher,student);remaining=len(result['primary'])+5-len(result['all_predictions'])
    for row in confirm_entries(entries):
        if not any(row['class_id']==p['class_id'] and row['box_xyxy']==p['box_xyxy'] and row['confidence']==p['confidence'] for p in allowed):continue
        if len(result['microcontext_additions'])>=remaining:break
        if any(iou(row['box_xyxy'],p['box_xyxy'])>=.5 for p in result['all_predictions']):continue
        row['student_microcontext_scores']=row.pop('teacher_context_view_scores')
        row['student_microcontext_windows']=row.pop('teacher_context_windows')
        row['evidence_tier']='student_multi_scale_context_manual_review'
        result['microcontext_additions'].append(row);result['all_predictions'].append(row)
    return result
